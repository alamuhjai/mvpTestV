import streamlit as st
import pandas as pd
import json

# ============================================================
# Safe JSON and Metric Handling
# ============================================================
def safe_json(x):
    """Safely parse JSON stored as string in CSV."""
    if pd.isna(x) or x == '':
        return {}
    try:
        return json.loads(x.replace("'", '"'))
    except:
        return {}

def safe_metric(val):
    """Ensure metric is numeric, default 0.0 if missing or NaN."""
    try:
        return float(val)
    except:
        return 0.0

# ============================================================
# Load Dataset (Student or Staff)
# ============================================================
@st.cache_data
def load_data(dataset="students"):
    try:
        path = "t5_final.csv" if dataset == "students" else "staff_diversity.csv"
        df = pd.read_csv(path)

        # Required columns
        if dataset == "students":
            required_columns = [
                'institution', 'city', 'state', 'level',
                'male_students', 'female_students', 'total_students',
                'gender_proportions', 'race_proportions',
                'descriptive_gender', 'descriptive_race', 'descriptive_joint',
                'representative_gender', 'representative_race', 'representative_joint',
                'compensatory_gender', 'compensatory_race', 'compensatory_joint',
                'blaus_gender', 'blaus_race',
                'R1', 'R2', 'D/PU', 'Masters', 'Baccalaureate', 'BacAssoc',
                'Associates', 'SpecialFocus', 'Tribal', 'HBCU', 'FaithRelated', 
                'MedicalHealth', 'EngineeringTech', 'Business', 'Arts', 'Law'
            ]
        else:
            required_columns = [
                'institution', 'city', 'state', 'level',
                'male_staff', 'female_staff', 'total_staff',
                'gender_proportions', 'race_proportions',
                'descriptive_gender', 'descriptive_race', 'descriptive_joint',
                'representative_gender', 'representative_race', 'representative_joint',
                'compensatory_gender', 'compensatory_race', 'compensatory_joint',
                'blaus_gender', 'blaus_race',
                'R2', 'D/PU', 'Masters', 'Baccalaureate', 'BacAssoc', 'Associates',
                'SpecialFocus', 'Tribal', 'HBCU', 'FaithRelated', 'MedicalHealth',
                'EngineeringTech', 'Business', 'Arts', 'Law'
            ]

        # Verify required columns exist
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Missing required column: {col}")
                st.stop()

        # Parse JSON columns safely
        for col in ["gender_proportions", "race_proportions"]:
            df[col] = df[col].apply(safe_json)

        # Calculate percentages
        df["percent_female"] = df["gender_proportions"].apply(lambda x: round(x.get("female", 0.0) * 100, 2))
        df["percent_of_color"] = df["race_proportions"].apply(lambda x: round((1.0 - x.get("white_nh", 0.0)) * 100, 2))

        # Sanitize all diversity metrics
        metrics = [
            'descriptive_gender', 'descriptive_race', 'descriptive_joint',
            'representative_gender', 'representative_race', 'representative_joint',
            'compensatory_gender', 'compensatory_race', 'compensatory_joint',
            'blaus_gender', 'blaus_race'
        ]
        for m in metrics:
            if m in df.columns:
                df[m] = df[m].apply(safe_metric)
            else:
                df[m] = 0.0

        # Standardize level column
        LEVEL_MAPPING = {
            'total': 'Total Student Body',
            'undergrad': 'Undergraduate',
            'undergraduate': 'Undergraduate',
            'grad': 'Graduate',
            'graduate': 'Graduate',
            'all': 'Total Student Body'
        }
        df["level"] = df["level"].astype(str).str.lower().map(LEVEL_MAPPING).fillna(df["level"])

        # Add US Census regions
        US_REGIONS = {
            "Northeast": ['CT','ME','MA','NH','RI','VT','NJ','NY','PA'],
            "Midwest": ['IL','IN','MI','OH','WI','IA','KS','MN','MO','NE','ND','SD'],
            "South": ['DE','FL','GA','MD','NC','SC','VA','DC','WV','AL','KY','MS','TN','AR','LA','OK','TX'],
            "West": ['AZ','CO','ID','MT','NV','NM','UT','WY','AK','CA','HI','OR','WA']
        }
        def map_region(state):
            for region, states in US_REGIONS.items():
                if state in states:
                    return region
            return "Other"
        df["region"] = df["state"].apply(map_region)

        return df

    except FileNotFoundError:
        st.error(f"CSV file not found: {path}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# ============================================================
# Carnegie Classification Options
# ============================================================
def get_carnegie_filters(dataset="students"):
    if dataset=="students":
        return {
            "R1: Doctoral Universities – Very high research": "R1",
            "R2: Doctoral Universities – High research": "R2",
            "D/PU: Doctoral/Professional Universities": "D/PU",
            "Master's Colleges and Universities": "Masters",
            "Baccalaureate Colleges": "Baccalaureate",
            "Baccalaureate/Associate's Colleges": "BacAssoc",
            "Associate's Colleges": "Associates",
            "Special Focus Institutions": "SpecialFocus",
            "Tribal Colleges and Universities": "Tribal",
            "HBCU (Historically Black Colleges)": "HBCU",
            "Faith-Related Institutions": "FaithRelated",
            "Medical Schools & Centers": "MedicalHealth",
            "Engineering and Technology Schools": "EngineeringTech",
            "Business & Management Schools": "Business",
            "Arts, Music & Design Schools": "Arts",
            "Law Schools": "Law"
        }
    else:
        return {
            "R2: Doctoral Universities – High research": "R2",
            "D/PU: Doctoral/Professional Universities": "D/PU",
            "Master's Colleges and Universities": "Masters",
            "Baccalaureate Colleges": "Baccalaureate",
            "Baccalaureate/Associate's Colleges": "BacAssoc",
            "Associate's Colleges": "Associates",
            "Special Focus Institutions": "SpecialFocus",
            "Tribal Colleges and Universities": "Tribal",
            "HBCU (Historically Black Colleges)": "HBCU",
            "Faith-Related Institutions": "FaithRelated",
            "Medical Schools & Centers": "MedicalHealth",
            "Engineering and Technology Schools": "EngineeringTech",
            "Business & Management Schools": "Business",
            "Arts, Music & Design Schools": "Arts",
            "Law Schools": "Law"
        }

# ============================================================
# Main App
# ============================================================
def main():
    st.set_page_config(layout="wide", page_title="Institutional Diversity Dashboard")
    st.title("🏛️ Institutional Diversity Dashboard")

    # Dataset selection
    dataset_type = st.radio("Select Dataset", ["Student Body", "Staff"])
    dataset_key = "students" if dataset_type == "Student Body" else "staff"
    df = load_data(dataset_key)
    carnegie_options = get_carnegie_filters(dataset_key)

    # ==============================
    # Sidebar Filters
    # ==============================
    # ==============================
    # Sidebar Filters
    # ==============================
    with st.sidebar:
        st.header("🔍 Filter Options")

        # Level filter
        # Level filter
        available_levels = sorted(df["level"].unique())

        # Determine default index
        if dataset_key == "students":
            # Make sure "Total" exists in available_levels
            default_level = "Total Student Body"
            if default_level in available_levels:
                default_index = available_levels.index(default_level)
            else:
                default_index = 0
        else:
            # For Staff, default to first level
            default_index = 0

        selected_level = st.radio(
            "Select Level",
            available_levels,
            index=default_index
        ) if available_levels else None


        # State filter
        selected_states = st.multiselect("Select States (optional)", sorted(df["state"].unique()))

        # Region filter
        US_REGIONS = {
            "Northeast": ['CT','ME','MA','NH','RI','VT','NJ','NY','PA'],
            "Midwest": ['IL','IN','MI','OH','WI','IA','KS','MN','MO','NE','ND','SD'],
            "South": ['DE','FL','GA','MD','NC','SC','VA','DC','WV','AL','KY','MS','TN','AR','LA','OK','TX'],
            "West": ['AZ','CO','ID','MT','NV','NM','UT','WY','AK','CA','HI','OR','WA']
        }
        selected_regions = st.multiselect("Select Regions (optional)", list(US_REGIONS.keys()))

        # Metric selection
        metric_options = {
            "Descriptive (Gender)": "descriptive_gender",
            "Descriptive (Race)": "descriptive_race",
            "Descriptive (Joint)": "descriptive_joint",
            "Representative (Gender)": "representative_gender",
            "Representative (Race)": "representative_race",
            "Representative (Joint)": "representative_joint",
            "Compensatory (Gender)": "compensatory_gender",
            "Compensatory (Race)": "compensatory_race",
            "Compensatory (Joint)": "compensatory_joint",
            "Blau Index (Gender)": "blaus_gender",
            "Blau Index (Race)": "blaus_race"
        }
        available_metrics = {k:v for k,v in metric_options.items() if v in df.columns}
        selected_metric_label = st.selectbox("Select Diversity Metric", list(available_metrics.keys()))
        selected_metric = available_metrics[selected_metric_label]

        # Carnegie Classification Filter
        st.subheader("🏫 Institution Types")
        carnegie_selections = {}
        for label, col in carnegie_options.items():
            if col in df.columns:
                carnegie_selections[col] = st.checkbox(label, value=True)


    # ==============================
    # Apply Filters
    # ==============================
    filtered_df = df.copy()
    if selected_level:
        filtered_df = filtered_df[filtered_df["level"] == selected_level]
    if selected_states:
        filtered_df = filtered_df[filtered_df["state"].isin(selected_states)]
    if selected_regions:
        region_states = [s for r in selected_regions for s in US_REGIONS[r]]
        filtered_df = filtered_df[filtered_df["state"].isin(region_states)]
    active_filters = [col for col, selected in carnegie_selections.items() if selected]
    if active_filters:
        filtered_df = filtered_df[filtered_df[active_filters].any(axis=1)]

    # ==============================
    # Empty Check and Summary Metrics
    # ==============================
    if filtered_df.empty:
        st.warning("No institutions match your filters.")
        st.stop()

    total_inst = len(filtered_df)
    avg_score = filtered_df[selected_metric].mean()
    top_score = filtered_df[selected_metric].max()


        
    st.header(f"📊 Ranking by {selected_metric_label}")
    
    # Create columns for summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_inst = len(filtered_df)
        st.metric("Total Institutions", total_inst)
    
    with col2:
        avg_score = filtered_df[selected_metric].mean()
        st.metric("Average Score", f"{avg_score:.3f}")
    
    with col3:
        top_score = filtered_df[selected_metric].max()
        st.metric("Top Score", f"{top_score:.3f}")
    
    with col4:
        # Show selected level
        if dataset_type == "Student Body":
            level_display = {
                'total': 'Total Student Body',
                'undergraduate': 'Undergraduate',
                'graduate': 'Graduate'
            }.get(selected_level, selected_level)
            st.metric("Student Level", level_display)
        else:
            st.metric("Staff Level", selected_level)

    # ==============================
    # Ranking Table
    # ==============================
    # ==============================
    # Ranking Table with Tie Handling
    # ==============================
    # Sort by selected metric (descending) and institution (ascending) as a tie-breaker
 
    # Ranking Table with Tie Handling
    # ==============================
    # Sort by selected metric (descending) and institution (ascending) as a tie-breaker
    filtered_df = filtered_df.sort_values(by=[selected_metric, "institution"], ascending=[False, True])

    # Rank using 'min' method so tied scores get the same rank
    filtered_df["rank"] = filtered_df[selected_metric].rank(method="min", ascending=False).astype(int)

    # Reset index so it goes from 1 to N (optional display)
    filtered_df = filtered_df.reset_index(drop=True)

    # Columns to display
    display_cols = ["rank", "institution", "city", "state", selected_metric, "percent_female", "percent_of_color"]

    if dataset_type == "Student Body":
        display_cols.insert(4, "level")
        display_cols.append("total_students")
        display_df = filtered_df[display_cols].rename(
            columns={selected_metric: "diversity_score", "total_students": "total_students", "level": "student_level"}
        )
    else:
        display_cols.append("total_staff")
        display_df = filtered_df[display_cols].rename(
            columns={selected_metric: "diversity_score", "total_staff": "total_staff"}
        )

    # Reset display index to start from 1
    display_df.index = range(1, len(display_df)+1)

    st.dataframe(display_df, use_container_width=True, height=600)


    # ==============================
    # General Dataset Info
    # ==============================
    with st.expander("📄 Data Information", expanded=False):
        if dataset_type=="Student Body":
            st.write(
                "This dataset contains student enrollment and demographic information for U.S. institutions. "
                "Includes gender and race distributions, total student counts, and diversity metrics."
            )
        else:
            st.write(
                "This dataset contains staff demographic information for U.S. institutions. "
                "Includes gender and race distributions, total staff counts, and diversity metrics."
            )

    # ==============================
    # Detailed Institution Info
    # ==============================
    with st.expander("🔍 Detailed Institution Information", expanded=False):
        inst_list = sorted(filtered_df["institution"].unique())
        inst = st.selectbox("Select Institution", inst_list)
        d = filtered_df[filtered_df["institution"]==inst].iloc[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Demographics")
            if dataset_type=="Student Body":
                st.metric("Total Students", f"{d['total_students']:,}")
                st.metric("Student Level", d['level'])
            else:
                st.metric("Total Staff", f"{d['total_staff']:,}")
                st.metric("Staff Level", d.get('level','N/A'))
            st.metric("Female %", f"{d['percent_female']:.1f}%")
            st.metric("People of Color %", f"{d['percent_of_color']:.1f}%")
        with col2:
            st.subheader("Primary Diversity Scores")
            st.metric("Selected Metric", f"{d[selected_metric]:.3f}")
            st.write(f"Descriptive Joint: {d['descriptive_joint']:.3f}")
            st.write(f"Representative Joint: {d['representative_joint']:.3f}")
            st.write(f"Compensatory Joint: {d['compensatory_joint']:.3f}")
        with col3:
            st.subheader("Blau's Indices")
            st.metric("Blau's Race Index", f"{d['blaus_race']:.3f}")
            st.metric("Blau's Gender Index", f"{d['blaus_gender']:.3f}")
            if dataset_type=="Student Body":
                classification = [cls for cls in ["R1","R2","HBCU","Tribal"] if d.get(cls,0)==1]
                if classification:
                    st.subheader("Classifications")
                    for cls in classification:
                        st.write(f"• {cls}")

    # ==============================
    # Extra Analytics (Average Scores)
    # ==============================
    with st.expander("📈 Extra Analytics", expanded=True):
        st.subheader("Average Diversity Score by State")
        avg_state = filtered_df.groupby("state")[selected_metric].agg(['mean','count'])
        avg_state = avg_state.rename(columns={'mean':'Average Score','count':'Institution Count'}).round(3)
        avg_state = avg_state.sort_values('Average Score', ascending=False)
        st.dataframe(avg_state, use_container_width=True, height=300)

        st.subheader("Average Diversity Score by Region")
        avg_region = filtered_df.groupby("region")[selected_metric].agg(['mean','count'])
        avg_region = avg_region.rename(columns={'mean':'Average Score','count':'Institution Count'}).round(3)
        avg_region = avg_region.sort_values('Average Score', ascending=False)
        st.dataframe(avg_region, use_container_width=True, height=204)

        st.subheader("Average Diversity Score by Institution Type")
        inst_types = [col for col in carnegie_options.values() if col in filtered_df.columns]
        type_scores = []
        for col in inst_types:
            df_type = filtered_df[filtered_df[col]==1]
            if not df_type.empty:
                type_scores.append({
                    "Institution Type": col,
                    "Average Score": df_type[selected_metric].mean().round(3),
                    "Count": len(df_type)
                })
        if type_scores:
            type_df = pd.DataFrame(type_scores).sort_values("Average Score", ascending=False)
            st.dataframe(type_df, use_container_width=True, height=300)

if __name__=="__main__":
    main()
