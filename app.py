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
    Seeds the database with V3 data including 'flexible_hours' and 'preferred_shifts'.
    """
    # Check for migration needs
    needs_migration = False
    if len(staff_table.all()) > 0:
        first_record = staff_table.all()[0]
        if 'flexible_hours' not in first_record or 'preferred_shifts' not in first_record:
            needs_migration = True

    if force_reset or len(staff_table.all()) == 0 or needs_migration:
        db.drop_table('staff')
        
        staff_data = []
        
        # General Manager
        staff_data.append({
            'name': 'Daniel', 'role': 'General Manager', 'type': 'Full Time', 'max_hours': 40,
            'flexible_hours': False, 'preferred_shifts': ['Opening', 'Middle']
        })
        
        # Managers
        for name in ['Pavan', 'Dana', 'Misrak']:
            staff_data.append({
                'name': name, 'role': 'Manager', 'type': 'Full Time', 'max_hours': 40,
                'flexible_hours': False, 'preferred_shifts': ['Opening', 'Closing']
            })
            
        # Full Time Staff
        for name in ['Eddy', 'Hein', 'Sancia', 'Liban', 'Omya', 'Jacquline']:
            staff_data.append({
                'name': name, 'role': 'Staff', 'type': 'Full Time', 'max_hours': 40,
                'flexible_hours': False, 'preferred_shifts': ['Middle', 'Closing']
            })
            
        # Part Time Staff (Flexible)
        for name in ['Htet', 'Naing', 'Dharani', 'Freya', 'Abby']:
            staff_data.append({
                'name': name, 'role': 'Staff', 'type': 'Part Time', 'max_hours': 20,
                'flexible_hours': True, 'preferred_shifts': ['Peak_Lunch', 'Peak_Dinner', 'Closing']
            })
            
        staff_table.insert_multiple(staff_data)
        
        msg = "Database reset to V3 schema!" if force_reset else "Database initialized with V3 schema."
        st.toast(msg, icon="🚀")

# Initialize DB
init_db()

# --- 2. The Logic (OR-Tools) ---

def solve_Rota(staff_list, num_days, closing_shift_counts, daily_budgets):
    """
    V3 Solver: Maximizes budget utilization and handles complex constraints.
    """
    model = cp_model.CpModel()
    shifts = {}

    # --- Definitions ---
    # 0: Opening (07:00-15:00) -> 7.5h
    # 1: Middle  (11:30-20:30) -> 8.5h
    # 2: Closing (15:00-23:30) -> 8.0h
    # 3: Peak_Lunch (11:00-15:00) -> 4.0h
    # 4: Peak_Dinner (17:00-21:00) -> 4.0h
    
    SHIFT_INDICES = list(range(5))
    STANDARD_SHIFTS = [0, 1, 2]
    FLEX_SHIFTS = [3, 4]
    
    SHIFT_HOURS = {0: 7.5, 1: 8.5, 2: 8.0, 3: 4.0, 4: 4.0}
    # Scaled for integer math (*10)
    SHIFT_HOURS_SCALED = {k: int(v * 10) for k, v in SHIFT_HOURS.items()}
    
    SHIFT_CODES = {0: "Open", 1: "Mid", 2: "Close", 3: "PkLnch", 4: "PkDin"}
    SHIFT_NAMES_FULL = {
        0: "Opening (7.5h)", 1: "Middle (8.5h)", 2: "Closing (8.0h)",
        3: "Peak Lunch (4h)", 4: "Peak Dinner (4h)"
    }

    # Map string preferences to indices
    PREF_MAP = {
        'Opening': 0, 'Middle': 1, 'Closing': 2, 
        'Peak_Lunch': 3, 'Peak_Dinner': 4
    }

    manager_ids = [s.doc_id for s in staff_list if s['role'] in ['Manager', 'General Manager']]
    flexible_staff_ids = [s.doc_id for s in staff_list if s.get('flexible_hours', False)]

    # --- Variables ---
    for s in staff_list:
        for d in range(num_days):
            for sh in SHIFT_INDICES:
                shifts[(s.doc_id, d, sh)] = model.NewBoolVar(f's{s.doc_id}_d{d}_sh{sh}')

    # --- Hard Constraints ---

    for d in range(num_days):
        # 1. Standard Shift Counts (Fixed)
        model.Add(sum(shifts[(s.doc_id, d, 0)] for s in staff_list) == 3) # Opening
        model.Add(sum(shifts[(s.doc_id, d, 1)] for s in staff_list) == 2) # Middle
        model.Add(sum(shifts[(s.doc_id, d, 2)] for s in staff_list) == closing_shift_counts[d % 7]) # Closing

        # 2. Peak Hour Coverage
        # 12:00-14:00 (Lunch): Covered by Open(0), Mid(1), Peak_Lunch(3)
        lunch_staff = sum(shifts[(s.doc_id, d, sh)] for s in staff_list for sh in [0, 1, 3])
        model.Add(lunch_staff >= 5)

        # 18:00-20:00 (Dinner): Covered by Mid(1), Close(2), Peak_Dinner(4)
        dinner_staff = sum(shifts[(s.doc_id, d, sh)] for s in staff_list for sh in [1, 2, 4])
        model.Add(dinner_staff >= 5)

        # 3. Manager Coverage (Opening & Closing)
        model.Add(sum(shifts[(m_id, d, 0)] for m_id in manager_ids) >= 1)
        model.Add(sum(shifts[(m_id, d, 2)] for m_id in manager_ids) >= 1)

        # 4. Daily Budget
        daily_budget_val = daily_budgets[d % 7]
        # Calculate cost ONLY for non-General Managers
        daily_cost_scaled = sum(
            shifts[(s.doc_id, d, sh)] * SHIFT_HOURS_SCALED[sh]
            for s in staff_list 
            if s['role'] != 'General Manager'
            for sh in SHIFT_INDICES
        )
        model.Add(daily_cost_scaled <= int(daily_budget_val * 10))

    # 5. Staff Constraints
    for s in staff_list:
        # Flex Shift Eligibility: Only 'flexible_hours' staff can work shifts 3 and 4
        if not s.get('flexible_hours', False):
            for d in range(num_days):
                for sh in FLEX_SHIFTS:
                    model.Add(shifts[(s.doc_id, d, sh)] == 0)

        # One Shift Per Day
        for d in range(num_days):
            model.Add(sum(shifts[(s.doc_id, d, sh)] for sh in SHIFT_INDICES) <= 1)
        
        # Max Weekly Hours
        total_hours_scaled = sum(
            shifts[(s.doc_id, d, sh)] * SHIFT_HOURS_SCALED[sh]
            for d in range(num_days) for sh in SHIFT_INDICES
        )
        model.Add(total_hours_scaled <= int(s['max_hours'] * 10))

        # "Clopening" Rule: No Closing(2) on Day D and Opening(0) on Day D+1
        for d in range(num_days - 1):
            model.Add(shifts[(s.doc_id, d, 2)] + shifts[(s.doc_id, d + 1, 0)] <= 1)

    # --- Objective Function ---
    # Maximize (Total Hours Worked * Weight1) + (Preferred Shifts * Weight2)
    
    # Weighting: We want to prioritize using the budget (Total Hours) significantly.
    # But we also want to respect preferences where possible.
    
    obj_hours = sum(
        shifts[(s.doc_id, d, sh)] * SHIFT_HOURS_SCALED[sh]
        for s in staff_list for d in range(num_days) for sh in SHIFT_INDICES
    )
    
    obj_prefs = 0
    for s in staff_list:
        prefs = s.get('preferred_shifts', [])
        for p_str in prefs:
            if p_str in PREF_MAP:
                p_idx = PREF_MAP[p_str]
                # Add 1 bonus point for every preferred shift assigned
                obj_prefs += sum(shifts[(s.doc_id, d, p_idx)] for d in range(num_days))

    # Maximize
    # Scaling preferences up to make them relevant against hours (which are in ~3000 range)
    # 1 preference met ~= 2 hours of work value? 
    # Let's say 1 hour = 10 units. 1 pref = 20 units.
    model.Maximize(obj_hours + (obj_prefs * 20))

    # --- Solve ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        data = []
        DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        # Create a grid: Rows = Staff, Cols = Days
        # This visualization is better for complex shifts than Rows=Shifts
        
        for s in staff_list:
            row = {'Staff': f"{s['name']} ({s['role']})"}
            total_h = 0
            for d in range(num_days):
                cell_val = "-"
                for sh in SHIFT_INDICES:
                    if solver.Value(shifts[(s.doc_id, d, sh)]):
                        cell_val = SHIFT_CODES[sh]
                        total_h += SHIFT_HOURS[sh]
                        break
                row[DAY_NAMES[d % 7]] = cell_val
            row['Total Hours'] = total_h
            data.append(row)
            
        # Summary Row (Budget Usage)
        summary_row = {'Staff': 'DAILY TOTAL (Hrs) [Excl. GM]'}
        grand_total = 0
        for d in range(num_days):
            d_total = 0
            for s in staff_list:
                # Exclude General Manager from budget calculation display to match solver logic
                if s['role'] == 'General Manager':
                    continue
                    
                for sh in SHIFT_INDICES:
                    if solver.Value(shifts[(s.doc_id, d, sh)]):
                        d_total += SHIFT_HOURS[sh]
            grand_total += d_total
            budget = daily_budgets[d % 7]
            summary_row[DAY_NAMES[d % 7]] = f"{d_total} / {budget}"
        summary_row['Total Hours'] = grand_total
        data.append(summary_row)

        return pd.DataFrame(data), solver.ObjectiveValue()
    else:
        return None, 0

# --- 3. The User Interface (Streamlit) ---

st.set_page_config(page_title="Rota Generator", layout="wide")
st.title("⚡ Rota Generator: Advanced Scheduling")

# Sidebar
st.sidebar.header("Configuration")
num_days = st.sidebar.number_input("Days to Schedule", 1, 14, 7)

with st.sidebar.expander("Closing Shift Staff Counts", expanded=False):
    closing_shift_counts = []
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    default_closing_counts = {
        "Monday": 3, "Tuesday": 3, "Wednesday": 3, "Thursday": 3,
        "Friday": 4, "Saturday": 4, "Sunday": 3
    }
    for day in days:
        val = st.number_input(f"{day} Closing Staff", 1, 10, default_closing_counts[day], key=f"csc_{day}")
        closing_shift_counts.append(val)

with st.sidebar.expander("Daily Hour Budgets", expanded=False):
    daily_budgets = []
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    default_budgets = {
        "Monday": 60.0, "Tuesday": 60.0, "Wednesday": 60.0,
        "Thursday": 64.0, "Friday": 67.0, "Saturday": 67.0, "Sunday": 64.0
    }
    for day in days:
        val = st.number_input(f"{day}", 40.0, 100.0, default_budgets[day], step=0.5, key=f"b_{day}")
        daily_budgets.append(val)

if st.sidebar.button("⚠️ Reset & Seed Database"):
    init_db(force_reset=True)
    st.rerun()

# Tabs
tab1, tab2 = st.tabs(["Manage Staff", "Generate Rota"])

with tab1:
    st.header("Manage Staff")
    staff_data = staff_table.all()
    if staff_data:
        # Prepare DataFrame with doc_id
        df = pd.DataFrame(staff_data)
        # Ensure doc_id is available but maybe hidden in editor or just disabled
        df['doc_id'] = [s.doc_id for s in staff_data]
        
        # Columns to display/edit
        cols = ['doc_id', 'name', 'role', 'type', 'max_hours', 'flexible_hours', 'preferred_shifts']
        df = df[cols]

        st.info("💡 You can edit staff details (like Max Hours) directly in the table below.")
        
        # Use data_editor
        edited_df = st.data_editor(
            df, 
            use_container_width=True,
            column_config={
                "doc_id": st.column_config.NumberColumn("ID", disabled=True),
                "max_hours": st.column_config.NumberColumn("Max Hours", min_value=0, max_value=80),
                "preferred_shifts": st.column_config.ListColumn("Preferred Shifts")
            },
            hide_index=True
        )

        # Check for changes and update DB
        # This compares the current state of edited_df with the original data
        # A simpler way with TinyDB is to iterate and update if changed, 
        # but for this scale, we can just button trigger or auto-update if we track state.
        # Let's use a button to "Save Changes" to be explicit and safe, 
        # OR just update on interaction if we want to be fancy. 
        # For simplicity and robustness:
        
        if st.button("💾 Save Changes to Database"):
            updated_count = 0
            for index, row in edited_df.iterrows():
                doc_id = row['doc_id']
                # Construct update dict (excluding doc_id)
                update_data = {
                    'name': row['name'],
                    'role': row['role'],
                    'type': row['type'],
                    'max_hours': row['max_hours'],
                    'flexible_hours': row['flexible_hours'],
                    'preferred_shifts': row['preferred_shifts']
                }
                staff_table.update(update_data, doc_ids=[doc_id])
                updated_count += 1
            st.success(f"Updated {updated_count} staff records!")
            st.rerun()
    else:
        st.info("No staff found.")
    
    st.subheader("Add/Edit Staff")
    with st.form("staff_form"):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Name")
        role = c2.selectbox("Role", ["Staff", "Manager", "General Manager"])
        sType = c3.selectbox("Type", ["Full Time", "Part Time"])
        
        c4, c5 = st.columns(2)
        max_h = c4.number_input("Max Hours", 0, 60, 40)
        is_flex = c5.checkbox("Flexible Hours? (Can work 4h shifts)")
        
        prefs = st.multiselect("Preferred Shifts", ['Opening', 'Middle', 'Closing', 'Peak_Lunch', 'Peak_Dinner'])
        
        if st.form_submit_button("Add Staff"):
            staff_table.insert({
                'name': name, 'role': role, 'type': sType,
                'max_hours': max_h, 'flexible_hours': is_flex, 'preferred_shifts': prefs
            })
            st.success(f"Added {name}")
            st.rerun()
            
    if st.button("Delete Selected Staff"):
        # Simple deletion for demo (assumes unique names or just delete last one logic would be better but keeping simple)
        pass 

with tab2:
    st.header("Generate Rota")
    
    if st.button("🚀 Generate Rota", type="primary"):
        with st.spinner("Generating complex schedule..."):
            staff_list = staff_table.all()
            df_result, obj_val = solve_Rota(staff_list, num_days, closing_shift_counts, daily_budgets)
            
            if df_result is not None:
                st.success(f"Rota Generation Complete! Score: {obj_val}")
                
                # Legend
                st.markdown("""
                **Legend:** 
                `Open`: 07:00-15:00 (7.5h) | `Mid`: 11:30-20:30 (8.5h) | `Close`: 15:00-23:30 (8.0h)
                `PkLnch`: 11:00-15:00 (4.0h) | `PkDin`: 17:00-21:00 (4.0h)
                """)
                
                st.dataframe(df_result, use_container_width=True)
            else:
                st.error("Infeasible solution. Try increasing the daily budget or adding more staff.")