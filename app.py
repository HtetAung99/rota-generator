import streamlit as st
from tinydb import TinyDB, Query
import pandas as pd
from ortools.sat.python import cp_model
import os

# --- 1. Data Storage & Initialization ---

DB_FILE = 'db.json'
db = TinyDB(DB_FILE)
staff_table = db.table('staff')

def init_db():
    """Seeds the database with dummy data if it's empty."""
    if len(staff_table.all()) == 0:
        dummy_data = [
            {'name': 'Alice', 'role': 'Manager', 'availability': 'All'},
            {'name': 'Bob', 'role': 'Server', 'availability': 'All'},
            {'name': 'Charlie', 'role': 'Chef', 'availability': 'All'},
            {'name': 'David', 'role': 'Manager', 'availability': 'All'},
            {'name': 'Eve', 'role': 'Server', 'availability': 'All'}
        ]
        staff_table.insert_multiple(dummy_data)
        st.toast("Database seeded with dummy data!", icon="🌱")

# Initialize DB on load
init_db()

# --- 3. The Logic (OR-Tools) ---

def solve_roster(staff_list, num_days=7, num_shifts=2):
    """
    Solves the scheduling problem using Google OR-Tools.
    Constraints:
    1. Exactly 1 person assigned to every shift.
    2. At least 1 Manager assigned per day (aggregated across shifts).
    """
    model = cp_model.CpModel()
    shifts = {}

    # staff_list is a list of TinyDB documents (dicts with doc_id)
    # Create variables: shifts[(staff_id, day, shift)]
    for s in staff_list:
        for d in range(num_days):
            for sh in range(num_shifts):
                shifts[(s.doc_id, d, sh)] = model.NewBoolVar(f'shift_s{s.doc_id}_d{d}_sh{sh}')

    # Constraint 1: Ensure exactly 1 person is assigned to every shift
    for d in range(num_days):
        for sh in range(num_shifts):
            model.Add(sum(shifts[(s.doc_id, d, sh)] for s in staff_list) == 1)

    # Constraint 2: Ensure at least 1 Manager is assigned per day
    managers = [s for s in staff_list if s.get('role', '').lower() == 'manager']
    
    # Only apply if we actually have managers, otherwise the problem is infeasible
    if managers:
        for d in range(num_days):
            # Sum of assignments for all managers across all shifts in a day >= 1
            model.Add(
                sum(shifts[(m.doc_id, d, sh)] for m in managers for sh in range(num_shifts)) >= 1
            )

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Format data for display
        # Rows = Shifts, Cols = Days
        schedule_data = []
        for sh in range(num_shifts):
            row = {}
            row['Shift'] = f"Shift {sh + 1}"
            for d in range(num_days):
                # Find who is working this shift
                worker_name = "Unassigned"
                for s in staff_list:
                    if solver.Value(shifts[(s.doc_id, d, sh)]):
                        worker_name = f"{s['name']} ({s['role']})"
                        break
                row[f"Day {d + 1}"] = worker_name
            schedule_data.append(row)
        
        return pd.DataFrame(schedule_data)
    else:
        return None

# --- 2. The User Interface (Streamlit) ---

st.set_page_config(page_title="Timetable Generator", layout="wide")
st.title("📅 Timetable Generator App")

# Sidebar
st.sidebar.header("Configuration")
num_days = st.sidebar.number_input("Days to Schedule", min_value=1, max_value=31, value=7)
num_shifts = st.sidebar.number_input("Shifts per Day", min_value=1, max_value=5, value=2)

if st.sidebar.button("Reset Database"):
    db.drop_table('staff')
    init_db()
    st.rerun()

# Tabs
tab1, tab2 = st.tabs(["Manage Staff", "Generate Roster"])

with tab1:
    st.header("Manage Staff")
    
    # Display Staff
    staff_data = staff_table.all()
    if staff_data:
        # Convert to DF for display, include doc_id for reference if needed (though hidden in simple view)
        df_staff = pd.DataFrame(staff_data)
        # Add a column for internal ID just for clarity in this raw view, or index by it
        df_staff['ID'] = [s.doc_id for s in staff_data]
        st.dataframe(df_staff.set_index('ID'), use_container_width=True)
    else:
        st.info("No staff found.")

    st.subheader("Add New Staff")
    with st.form("add_staff_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        new_name = col1.text_input("Name")
        new_role = col2.selectbox("Role", ["Server", "Chef", "Manager", "Bartender"])
        submitted = st.form_submit_button("Add Staff")
        
        if submitted and new_name:
            staff_table.insert({'name': new_name, 'role': new_role, 'availability': 'All'})
            st.success(f"Added {new_name}")
            st.rerun()

    st.subheader("Delete Staff")
    if staff_data:
        staff_options = {f"{s.doc_id}: {s['name']}": s.doc_id for s in staff_data}
        selected_staff_label = st.selectbox("Select Staff to Delete", options=list(staff_options.keys()))
        
        if st.button("Delete Selected"):
            doc_id_to_delete = staff_options[selected_staff_label]
            staff_table.remove(doc_ids=[doc_id_to_delete])
            st.warning(f"Deleted staff ID {doc_id_to_delete}")
            st.rerun()

with tab2:
    st.header("Generate Roster")
    st.write(f"Generating schedule for **{num_days} days** with **{num_shifts} shifts/day**.")
    
    if st.button("🚀 Generate Schedule", type="primary"):
        with st.spinner("Running OR-Tools Solver..."):
            staff_list = staff_table.all()
            
            if not staff_list:
                st.error("No staff available to schedule! Please add staff in the 'Manage Staff' tab.")
            else:
                result_df = solve_roster(staff_list, num_days=num_days, num_shifts=num_shifts)
                
                if result_df is not None:
                    st.success("Schedule Generated Successfully!")
                    st.dataframe(result_df, hide_index=True, use_container_width=True)
                else:
                    st.error("Could not find a feasible schedule. Try adding more staff or ensuring there is at least one Manager.")
