import streamlit as st
from tinydb import TinyDB, Query
import pandas as pd
from ortools.sat.python import cp_model
import os

# --- 1. Data Storage & Initialization ---

DB_FILE = 'db.json'
db = TinyDB(DB_FILE)
staff_table = db.table('staff')

def init_db(force_reset=False):
    """
    Seeds the database with specific staff members and roles.
    New structure includes 'employment_type' and 'max_hours_week'.
    """
    needs_migration = False
    if len(staff_table.all()) > 0:
        first_record = staff_table.all()[0]
        if 'type' not in first_record or 'max_hours' not in first_record:
            needs_migration = True

    if force_reset or len(staff_table.all()) == 0 or needs_migration:
        db.drop_table('staff')
        
        # Define seed data
        # Defaults: FT = 40h, PT = 20h
        staff_data = []
        
        # General Manager
        staff_data.append({'name': 'Daniel', 'role': 'General Manager', 'type': 'Full Time', 'max_hours': 40})
        
        # Managers
        for name in ['Pavan', 'Dana', 'Misrak']:
            staff_data.append({'name': name, 'role': 'Manager', 'type': 'Full Time', 'max_hours': 40})
            
        # Full Time Staff
        for name in ['Eddy', 'Hein', 'Sancia', 'Liban', 'Omya', 'Jacquline']:
            staff_data.append({'name': name, 'role': 'Staff', 'type': 'Full Time', 'max_hours': 40})
            
        # Part Time Staff
        for name in ['Htet', 'Naing', 'Dharani', 'Freya', 'Abby']:
            staff_data.append({'name': name, 'role': 'Staff', 'type': 'Part Time', 'max_hours': 20})
            
        staff_table.insert_multiple(staff_data)
        if force_reset:
            st.toast("Database reset and seeded with V2 data!", icon="🔄")
        else:
            st.toast("Database initialized with V2 data.", icon="🌱")

# Initialize DB on load (only if empty)
init_db()

# --- 2. The Logic (OR-Tools) ---

def solve_roster(staff_list, num_days, closing_shift_count, daily_budgets):
    """
    Solves the scheduling problem using Google OR-Tools with V2 constraints.
    
    Shift Definitions (Working Hours):
    0: Opening (07:00-15:00) -> 7.5h
    1: Middle  (11:30-20:30) -> 8.5h
    2: Closing (15:00-23:30) -> 8.0h
    
    Peak Coverage Note:
    - 12:00-14:00 is covered by Opening + Middle (3 + 2 = 5 staff).
    - 18:00-20:00 is covered by Middle + Closing (2 + 3+ = 5+ staff).
    """
    model = cp_model.CpModel()
    shifts = {}
    
    # Constants for integer scaling (OR-Tools requires integers)
    # Scaling factor 10: 7.5h -> 75, 8.5h -> 85, 8.0h -> 80
    SHIFT_HOURS_SCALED = {0: 75, 1: 85, 2: 80}
    SHIFT_NAMES = {0: "Opening (07:00-15:00)", 1: "Middle (11:30-20:30)", 2: "Closing (15:00-23:30)"}
    NUM_SHIFTS = 3
    
    # Identify Managers (Manager or General Manager)
    manager_ids = [s.doc_id for s in staff_list if s['role'] in ['Manager', 'General Manager']]

    # Create variables: shifts[(staff_id, day, shift_idx)]
    for s in staff_list:
        for d in range(num_days):
            for sh in range(NUM_SHIFTS):
                shifts[(s.doc_id, d, sh)] = model.NewBoolVar(f'shift_s{s.doc_id}_d{d}_sh{sh}')

    # --- Constraints ---

    # 1. Fixed Staff Counts Per Shift
    for d in range(num_days):
        # Opening: Always 3
        model.Add(sum(shifts[(s.doc_id, d, 0)] for s in staff_list) == 3)
        # Middle: Always 2
        model.Add(sum(shifts[(s.doc_id, d, 1)] for s in staff_list) == 2)
        # Closing: Parameterized (Default 3)
        model.Add(sum(shifts[(s.doc_id, d, 2)] for s in staff_list) == closing_shift_count)

    # 2. Manager Coverage Rule
    # Opening: >= 1 Manager
    # Closing: >= 1 Manager
    # Middle: No specific requirement
    for d in range(num_days):
        model.Add(sum(shifts[(m_id, d, 0)] for m_id in manager_ids) >= 1)
        model.Add(sum(shifts[(m_id, d, 2)] for m_id in manager_ids) >= 1)

    # 3. One Shift Per Day
    for s in staff_list:
        for d in range(num_days):
            model.Add(sum(shifts[(s.doc_id, d, sh)] for sh in range(NUM_SHIFTS)) <= 1)

    # 4. Weekly Hour Limits (Part-Time/Full-Time Constraints)
    # Using scaled hours
    for s in staff_list:
        max_hours_scaled = int(s['max_hours'] * 10)
        total_hours_expr = sum(
            shifts[(s.doc_id, d, sh)] * SHIFT_HOURS_SCALED[sh]
            for d in range(num_days)
            for sh in range(NUM_SHIFTS)
        )
        model.Add(total_hours_expr <= max_hours_scaled)

    # 5. Daily Budget Constraint (Per Day)
    for d in range(num_days):
        budget_for_day = daily_budgets[d % 7]  # Use modulo to map day index to Monday-Sunday inputs
        max_daily_budget_scaled = int(budget_for_day * 10)
        
        daily_hours_expr = sum(
            shifts[(s.doc_id, d, sh)] * SHIFT_HOURS_SCALED[sh]
            for s in staff_list
            for sh in range(NUM_SHIFTS)
        )
        model.Add(daily_hours_expr <= max_daily_budget_scaled)

    # --- Solve ---
    solver = cp_model.CpSolver()
    # Optional: Set a time limit for complex solves
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Format output
        data = []
        DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for sh in range(NUM_SHIFTS):
            row = {'Shift': SHIFT_NAMES[sh]}
            for d in range(num_days):
                # Gather all workers for this shift/day
                workers = []
                for s in staff_list:
                    if solver.Value(shifts[(s.doc_id, d, sh)]):
                        role_short = "M" if s['role'] in ['Manager', 'General Manager'] else "S"
                        workers.append(f"{s['name']}({role_short})")
                
                # Join with commas/newlines for display
                # Use modulo to cycle through day names if num_days > 7, though current UI max is 14.
                row[DAY_NAMES[d % 7]] = ", ".join(workers) 
            data.append(row)
        
        # Add Daily Total Hours Row
        total_row = {'Shift': 'Total Hours'}
        for d in range(num_days):
            daily_total = 0.0
            for sh in range(NUM_SHIFTS):
                # Count assigned staff for this shift/day
                count = sum(solver.Value(shifts[(s.doc_id, d, sh)]) for s in staff_list)
                # Map shift index to hours (unscaled)
                shift_hours = {0: 7.5, 1: 8.5, 2: 8.0}[sh]
                daily_total += count * shift_hours
            total_row[DAY_NAMES[d % 7]] = f"{daily_total}h"
        data.append(total_row)

        return pd.DataFrame(data)
    else:
        return None

# --- 3. The User Interface (Streamlit) ---

st.set_page_config(page_title="Rota Generator V2", layout="wide")
st.title("📅 Rota Generator: Advanced Scheduling")

# Sidebar Configuration
st.sidebar.header("Configuration")
num_days = st.sidebar.number_input("Days to Schedule", min_value=1, max_value=14, value=7)

st.sidebar.subheader("Shift Parameters")
closing_staff_count = st.sidebar.number_input("Closing Shift Staff Count", min_value=1, value=3, help="Default is 3. Increase for weekends.")

st.sidebar.divider()
st.sidebar.subheader("Daily Budget Configuration")
st.sidebar.write("Set max hours for each day:")

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
daily_budgets = []

# Use an expander to keep the sidebar clean, or just list them if it fits
with st.sidebar.expander("Daily Hour Budgets", expanded=True):
    for day in DAY_NAMES:
        # Default value was 70.0 in previous version
        val = st.number_input(f"{day} Budget", min_value=40.0, value=70.0, step=0.5, key=f"budget_{day}")
        daily_budgets.append(val)

st.sidebar.divider()
if st.sidebar.button("⚠️ Reset & Seed Database"):
    init_db(force_reset=True)
    st.rerun()

# Tabs
tab1, tab2 = st.tabs(["Manage Staff", "Generate Rota"])

with tab1:
    st.header("Manage Staff")
    
    staff_data = staff_table.all()
    if staff_data:
        df_staff = pd.DataFrame(staff_data)
        # Reorder columns for better readability
        cols = ['name', 'role', 'type', 'max_hours']
        df_staff = df_staff[cols]
        st.dataframe(df_staff, use_container_width=True)
    else:
        st.info("No staff found.")

    st.subheader("Add New Staff")
    with st.form("add_staff_form", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns(4)
        n_name = c1.text_input("Name")
        n_role = c2.selectbox("Role", ["Staff", "Manager", "General Manager"])
        n_type = c3.selectbox("Type", ["Full Time", "Part Time"])
        n_hours = c4.number_input("Max Hours/Week", min_value=0, value=40 if n_type == "Full Time" else 20)
        
        if st.form_submit_button("Add Staff"):
            if n_name:
                staff_table.insert({
                    'name': n_name, 
                    'role': n_role, 
                    'type': n_type, 
                    'max_hours': n_hours
                })
                st.success(f"Added {n_name}")
                st.rerun()

    st.subheader("Delete Staff")
    if staff_data:
        staff_options = {f"{s.doc_id}: {s['name']}": s.doc_id for s in staff_data}
        sel_del = st.selectbox("Select Staff to Delete", options=list(staff_options.keys()))
        if st.button("Delete Selected"):
            staff_table.remove(doc_ids=[staff_options[sel_del]])
            st.rerun()

with tab2:
    st.header("Generate Roster")
    
    st.markdown("""
    **Logic Overview:**
    - **Opening (07:00-15:00):** 3 Staff (Must include 1 Manager)
    - **Middle (11:30-20:30):** 2 Staff
    - **Closing (15:00-23:30):** Variable (Default 3, Must include 1 Manager)
    
    *Peak Hours (12-2pm & 6-8pm) are covered by the natural overlap of shifts.*
    """)
    
    if st.button("🚀 Generate Schedule", type="primary"):
        with st.spinner("Optimizing schedule..."):
            staff_list = staff_table.all()
            if not staff_list:
                st.error("No staff available!")
            else:
                result = solve_roster(staff_list, num_days, closing_staff_count, daily_budgets)
                
                if result is not None:
                    st.success("Optimization Successful!")
                    st.dataframe(result, hide_index=True, use_container_width=True)
                else:
                    st.error("Infeasible! Try increasing the Daily Budgets or adjusting constraints.")
