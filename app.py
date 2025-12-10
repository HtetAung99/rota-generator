import streamlit as st
from tinydb import TinyDB, Query
import pandas as pd
from ortools.sat.python import cp_model
import os
import json
from datetime import datetime, timedelta

# --- 1. Data Storage & Initialization ---

DB_FILE = 'db.json'

def get_db():
    """Safely get the TinyDB instance, handling corruption."""
    try:
        return TinyDB(DB_FILE)
    except Exception:
        # If DB is corrupted, remove it and return a new instance
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
        return TinyDB(DB_FILE)

db = get_db()
staff_table = db.table('staff')
requests_table = db.table('requests')

def init_db(force_reset=False):
    """
    Seeds the database with V3 data including 'flexible_hours' and 'preferred_shifts'.
    Handles corrupted JSON files by resetting.
    """
    global db, staff_table
    
    try:
        # Check for migration needs
        needs_migration = False
        if len(staff_table.all()) > 0:
            first_record = staff_table.all()[0]
            if 'flexible_hours' not in first_record or 'preferred_shifts' not in first_record:
                needs_migration = True

        if force_reset or len(staff_table.all()) == 0 or needs_migration:
            db.drop_table('staff')
            db.drop_table('requests') # Clear requests on reset too
            
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
            
    except Exception as e:
        st.error(f"Database error detected: {e}. Resetting database...")
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
        # Re-initialize globals
        db = TinyDB(DB_FILE)
        staff_table = db.table('staff')
        init_db(force_reset=True)

# Initialize DB
init_db()

# --- 2. The Logic (OR-Tools) ---

def solve_Rota(staff_list, num_days, closing_shift_counts, daily_budgets, start_date, requests_list):
    """
    V3 Solver: Maximizes budget utilization, handles complex constraints & User Requests.
    """
    model = cp_model.CpModel()
    shifts = {}

    # --- Definitions ---
    SHIFT_INDICES = list(range(5))
    FLEX_SHIFTS = [3, 4]
    
    SHIFT_HOURS = {0: 7.5, 1: 8.5, 2: 8.0, 3: 4.0, 4: 4.0}
    # Scaled for integer math (*10)
    SHIFT_HOURS_SCALED = {k: int(v * 10) for k, v in SHIFT_HOURS.items()}
    
    SHIFT_CODES = {0: "Open", 1: "Mid", 2: "Close", 3: "PkLnch", 4: "PkDin"}
    SHIFT_NAMES_MAP = {
        "Opening": 0, "Middle": 1, "Closing": 2, 
        "Peak Lunch": 3, "Peak Dinner": 4
    }
    
    # Map string preferences to indices
    PREF_MAP = {
        'Opening': 0, 'Middle': 1, 'Closing': 2, 
        'Peak_Lunch': 3, 'Peak_Dinner': 4
    }

    manager_ids = [s.doc_id for s in staff_list if s['role'] in ['Manager', 'General Manager']]
    
    # Generate mapping for dates and weekdays
    # date_to_day_idx: '2025-12-25' -> 3
    # weekday_to_day_indices: 'Monday' -> [0, 7] (if 14 day schedule)
    
    date_to_day_idx = {}
    weekday_to_day_indices = {k: [] for k in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}
    
    current_dt = start_date
    for d in range(num_days):
        d_str = current_dt.strftime("%Y-%m-%d")
        w_str = current_dt.strftime("%A")
        
        date_to_day_idx[d_str] = d
        weekday_to_day_indices[w_str].append(d)
        
        current_dt += timedelta(days=1)

    # --- Variables ---
    for s in staff_list:
        for d in range(num_days):
            for sh in SHIFT_INDICES:
                shifts[(s.doc_id, d, sh)] = model.NewBoolVar(f's{s.doc_id}_d{d}_sh{sh}')

    # --- User Requests (Hard Rules) ---
    for req in requests_list:
        s_name = req['staff_name']
        # Find staff doc_id
        staff_obj = next((s for s in staff_list if s['name'] == s_name), None)
        if not staff_obj:
            continue
        s_id = staff_obj.doc_id
        
        req_type = req['request_type']
        val = req['value']
        
        # 1. OFF_SPECIFIC_DATE
        if req_type == 'OFF_SPECIFIC_DATE':
            if val in date_to_day_idx:
                d = date_to_day_idx[val]
                model.Add(sum(shifts[(s_id, d, sh)] for sh in SHIFT_INDICES) == 0)
                
        # 2. OFF_RECURRING_DAY
        elif req_type == 'OFF_RECURRING_DAY':
            if val in weekday_to_day_indices:
                for d in weekday_to_day_indices[val]:
                    model.Add(sum(shifts[(s_id, d, sh)] for sh in SHIFT_INDICES) == 0)

        # 3. WORK_SPECIFIC_SHIFT (Specific Date)
        elif req_type == 'WORK_SPECIFIC_SHIFT':
            # Value format expected: "2025-12-25 | Opening"
            try:
                date_part, shift_name = val.split(" | ")
                if date_part in date_to_day_idx and shift_name in SHIFT_NAMES_MAP:
                    d = date_to_day_idx[date_part]
                    sh_idx = SHIFT_NAMES_MAP[shift_name]
                    model.Add(shifts[(s_id, d, sh_idx)] == 1)
            except ValueError:
                pass # Invalid format ignored

        # 4. WORK_RECURRING_SHIFT
        elif req_type == 'WORK_RECURRING_SHIFT':
            # Value format expected: "Monday | Opening"
            try:
                day_name, shift_name = val.split(" | ")
                if day_name in weekday_to_day_indices and shift_name in SHIFT_NAMES_MAP:
                    sh_idx = SHIFT_NAMES_MAP[shift_name]
                    for d in weekday_to_day_indices[day_name]:
                        model.Add(shifts[(s_id, d, sh_idx)] == 1)
            except ValueError:
                pass

    # --- Hard Constraints ---

    for d in range(num_days):
        # 1. Standard Shift Counts (Fixed)
        model.Add(sum(shifts[(s.doc_id, d, 0)] for s in staff_list) == 3) # Opening
        model.Add(sum(shifts[(s.doc_id, d, 1)] for s in staff_list) == 2) # Middle
        model.Add(sum(shifts[(s.doc_id, d, 2)] for s in staff_list) == closing_shift_counts[d % 7]) # Closing

        # 2. Peak Hour Coverage
        lunch_staff = sum(shifts[(s.doc_id, d, sh)] for s in staff_list for sh in [0, 1, 3])
        model.Add(lunch_staff >= 5)

        dinner_staff = sum(shifts[(s.doc_id, d, sh)] for s in staff_list for sh in [1, 2, 4])
        model.Add(dinner_staff >= 5)

        # 3. Manager Coverage
        model.Add(sum(shifts[(m_id, d, 0)] for m_id in manager_ids) >= 1)
        model.Add(sum(shifts[(m_id, d, 2)] for m_id in manager_ids) >= 1)

        # 4. Daily Budget (Excl. GM)
        daily_budget_val = daily_budgets[d % 7]
        daily_cost_scaled = sum(
            shifts[(s.doc_id, d, sh)] * SHIFT_HOURS_SCALED[sh]
            for s in staff_list 
            if s['role'] != 'General Manager'
            for sh in SHIFT_INDICES
        )
        model.Add(daily_cost_scaled <= int(daily_budget_val * 10))

    # 5. Staff Constraints
    for s in staff_list:
        # Flex Eligibility
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

        # "Clopening" Rule
        for d in range(num_days - 1):
            model.Add(shifts[(s.doc_id, d, 2)] + shifts[(s.doc_id, d + 1, 0)] <= 1)

    # --- Objective Function ---
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
                obj_prefs += sum(shifts[(s.doc_id, d, p_idx)] for d in range(num_days))

    model.Maximize(obj_hours + (obj_prefs * 20))

    # --- Solve ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        data = []
        
        # Build list of day labels for the columns
        # e.g. "Mon 25 Dec"
        day_headers = []
        curr = start_date
        for _ in range(num_days):
            day_headers.append(curr.strftime("%a %d"))
            curr += timedelta(days=1)
            
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
                row[day_headers[d]] = cell_val
            row['Total Hours'] = total_h
            data.append(row)
            
        # Summary Row
        summary_row = {'Staff': 'DAILY TOTAL (Hrs) [Excl. GM]'}
        grand_total = 0
        for d in range(num_days):
            d_total = 0
            for s in staff_list:
                if s['role'] == 'General Manager': continue
                for sh in SHIFT_INDICES:
                    if solver.Value(shifts[(s.doc_id, d, sh)]):
                        d_total += SHIFT_HOURS[sh]
            grand_total += d_total
            budget = daily_budgets[d % 7]
            summary_row[day_headers[d]] = f"{d_total} / {budget}"
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

# Calculate next Monday
today = datetime.now().date()
days_until_monday = (0 - today.weekday() + 7) % 7 # Monday is weekday 0
next_monday = today + timedelta(days=days_until_monday)

st.sidebar.markdown(f"**Schedule for week starting:** `{next_monday.strftime('%A, %Y-%m-%d')}`")
start_date_dt = datetime.combine(next_monday, datetime.min.time()) # Ensure it's a datetime object

num_days = 7 # Always schedule for 7 days
st.sidebar.caption(f"Scheduling for {num_days} days automatically.")

with st.sidebar.expander("Closing Shift Staff Counts", expanded=False):
    closing_shift_counts = []
    days_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    default_closing_counts = {
        "Monday": 3, "Tuesday": 3, "Wednesday": 3, "Thursday": 3,
        "Friday": 4, "Saturday": 4, "Sunday": 3
    }
    for day in days_names:
        val = st.number_input(f"{day} Closing Staff", 1, 10, default_closing_counts[day], key=f"csc_{day}")
        closing_shift_counts.append(val)

with st.sidebar.expander("Daily Hour Budgets", expanded=False):
    daily_budgets = []
    default_budgets = {
        "Monday": 60.0, "Tuesday": 60.0, "Wednesday": 60.0,
        "Thursday": 64.0, "Friday": 67.0, "Saturday": 67.0, "Sunday": 64.0
    }
    for day in days_names:
        val = st.number_input(f"{day}", 40.0, 100.0, default_budgets[day], step=0.5, key=f"b_{day}")
        daily_budgets.append(val)

if st.sidebar.button("⚠️ Reset & Seed Database"):
    init_db(force_reset=True)
    st.rerun()

# Tabs
tab1, tab2, tab3 = st.tabs(["Manage Staff", "Requests & Availability", "Generate Rota"])

# --- TAB 1: Manage Staff ---
with tab1:
    st.header("Manage Staff")
    staff_data = staff_table.all()
    if staff_data:
        df = pd.DataFrame(staff_data)
        df['doc_id'] = [s.doc_id for s in staff_data]
        cols = ['doc_id', 'name', 'role', 'type', 'max_hours', 'flexible_hours', 'preferred_shifts']
        df = df[cols]

        st.info("💡 You can edit staff details directly in the table below.")
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

        if st.button("💾 Save Changes to Database"):
            updated_count = 0
            for index, row in edited_df.iterrows():
                doc_id = row['doc_id']
                update_data = {
                    'name': row['name'], 'role': row['role'], 'type': row['type'],
                    'max_hours': row['max_hours'], 'flexible_hours': row['flexible_hours'],
                    'preferred_shifts': row['preferred_shifts']
                }
                staff_table.update(update_data, doc_ids=[doc_id])
                updated_count += 1
            st.success(f"Updated {updated_count} staff records!")
            st.rerun()
    else:
        st.info("No staff found.")
    
    st.divider()
    st.subheader("Add New Staff")
    with st.form("staff_form"):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Name")
        role = c2.selectbox("Role", ["Staff", "Manager", "General Manager"])
        sType = c3.selectbox("Type", ["Full Time", "Part Time"])
        c4, c5 = st.columns(2)
        max_h = c4.number_input("Max Hours", 0, 60, 40)
        is_flex = c5.checkbox("Flexible Hours?")
        prefs = st.multiselect("Preferred Shifts", ['Opening', 'Middle', 'Closing', 'Peak_Lunch', 'Peak_Dinner'])
        
        if st.form_submit_button("Add Staff"):
            staff_table.insert({
                'name': name, 'role': role, 'type': sType,
                'max_hours': max_h, 'flexible_hours': is_flex, 'preferred_shifts': prefs
            })
            st.success(f"Added {name}")
            st.rerun()

# --- TAB 2: Requests ---
with tab2:
    st.header("Requests & Availability")
    
    st.subheader("Add New Request")
    
    # 1. Inputs outside form to allow interactivity
    staff_names = [s['name'] for s in staff_table.all()]
    
    c1, c2 = st.columns(2)
    r_staff = c1.selectbox("Staff Member", staff_names)
    r_type = c2.selectbox("Request Type", [
        "OFF_SPECIFIC_DATE", "OFF_RECURRING_DAY", 
        "WORK_SPECIFIC_SHIFT", "WORK_RECURRING_SHIFT"
    ])
    
    # 2. Dynamic inputs based on type (Immediate update)
    r_value = ""
    days_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    shifts_list = ["Opening", "Middle", "Closing", "Peak Lunch", "Peak Dinner"]
    
    # Use a container for the conditional inputs to keep layout clean
    with st.container():
        c3, c4 = st.columns(2)
        
        if r_type == "OFF_SPECIFIC_DATE":
            date_val = c3.date_input("Select Date", min_value=today)
            r_value = date_val.strftime("%Y-%m-%d")
            
        elif r_type == "OFF_RECURRING_DAY":
            day_val = c3.selectbox("Select Day", days_list)
            r_value = day_val
            
        elif r_type == "WORK_SPECIFIC_SHIFT":
            date_val = c3.date_input("Select Date", min_value=today)
            shift_val = c4.selectbox("Select Shift", shifts_list)
            r_value = f"{date_val.strftime('%Y-%m-%d')} | {shift_val}"
            
        elif r_type == "WORK_RECURRING_SHIFT":
            day_val = c3.selectbox("Select Day", days_list)
            shift_val = c4.selectbox("Select Shift", shifts_list)
            r_value = f"{day_val} | {shift_val}"
            
    # 3. Add Button
    if st.button("Add Rule", type="primary"):
        requests_table.insert({
            'staff_name': r_staff,
            'request_type': r_type,
            'value': r_value
        })
        st.success("Rule Added!")
        st.rerun()
            
    # Display Active Rules
    st.subheader("Active Rules")
    reqs = requests_table.all()
    if reqs:
        # Show as simple table with Delete button
        for r in reqs:
            col1, col2 = st.columns([4, 1])
            with col1:
                display_value = r['value']
                if r['request_type'] in ['OFF_SPECIFIC_DATE', 'WORK_SPECIFIC_SHIFT']:
                    try:
                        # Extract date part. For WORK_SPECIFIC_SHIFT it's "YYYY-MM-DD | Shift"
                        date_str_part = r['value'].split(" | ")[0]
                        date_obj = datetime.strptime(date_str_part, "%Y-%m-%d").date()
                        day_name = date_obj.strftime("%A")
                        # Reconstruct display value
                        if r['request_type'] == 'OFF_SPECIFIC_DATE':
                            display_value = f"{date_str_part} ({day_name})"
                        else: # WORK_SPECIFIC_SHIFT
                            shift_name = r['value'].split(" | ")[1]
                            display_value = f"{date_str_part} ({day_name}) | {shift_name}"
                    except (ValueError, IndexError):
                        # Fallback if date parsing or split fails
                        pass
                st.write(f"**{r['staff_name']}** - {r['request_type']}: `{display_value}`")
            with col2:
                if st.button("Delete", key=f"del_{r.doc_id}"):
                    requests_table.remove(doc_ids=[r.doc_id])
                    st.rerun()
    else:
        st.info("No active requests.")

# --- TAB 3: Generate Rota ---
with tab3:
    st.header("Generate Rota")
    
    if st.button("🚀 Generate Rota", type="primary"):
        with st.spinner("Generating complex schedule..."):
            staff_list = staff_table.all()
            requests_list = requests_table.all()
            
            # Pass start_date and requests to solver
            df_result, obj_val = solve_Rota(
                staff_list, num_days, closing_shift_counts, daily_budgets, 
                start_date_dt, requests_list
            )
            
            if df_result is not None:
                st.success(f"Rota Generation Complete! Score: {obj_val}")
                
                st.markdown("""
                **Legend:** 
                `Open`: 07:00-15:00 (7.5h) | `Mid`: 11:30-20:30 (8.5h) | `Close`: 15:00-23:30 (8.0h)
                `PkLnch`: 11:00-15:00 (4.0h) | `PkDin`: 17:00-21:00 (4.0h)
                """)
                
                st.dataframe(df_result, use_container_width=True)
            else:
                st.error("Infeasible solution. Check for conflicting requests or budget constraints.")
