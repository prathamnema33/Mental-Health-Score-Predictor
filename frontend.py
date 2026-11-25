import json
import os
from math import cos, radians, sin
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Circle, Wedge

PREDICTION_API_URL = os.environ.get(
    "PREDICTION_API_URL", "http://127.0.0.1:8000/predict"
)


def get_prediction(payload):
    request = Request(
        PREDICTION_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        result = json.loads(response.read().decode("utf-8"))

    if not isinstance(result, dict):
        raise ValueError("The prediction API returned an invalid response.")

    score = result.get("predict_mental_health_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("The prediction API response did not contain a valid score.")
    return float(score)


def render_score_gauge(score):
    score_for_gauge = min(max(score, 0.0), 10.0)
    score_angle = 180 - (score_for_gauge / 10.0) * 180

    figure, axis = plt.subplots(figsize=(5, 3))
    axis.add_patch(
        Wedge(
            (0, 0),
            1,
            0,
            180,
            width=0.18,
            facecolor="#E5E7EB",
            edgecolor="none",
        )
    )
    if score_for_gauge > 0:
        axis.add_patch(
            Wedge(
                (0, 0),
                1,
                score_angle,
                180,
                width=0.18,
                facecolor="#14B8A6",
                edgecolor="none",
            )
        )

    needle_x = 0.78 * cos(radians(score_angle))
    needle_y = 0.78 * sin(radians(score_angle))
    axis.plot([0, needle_x], [0, needle_y], color="#1F2937", linewidth=3)
    axis.add_patch(Circle((0, 0), 0.055, color="#1F2937"))
    axis.text(0, 0.28, f"{score:.2f}", ha="center", va="center", fontsize=20)
    axis.text(-1.02, -0.08, "0", ha="center", va="top")
    axis.text(1.02, -0.08, "10", ha="center", va="top")
    axis.set_aspect("equal")
    axis.set_xlim(-1.12, 1.12)
    axis.set_ylim(-0.2, 1.12)
    axis.axis("off")

    st.pyplot(figure, width="stretch")
    plt.close(figure)


st.set_page_config(layout="wide")
st.title('Mental Health Score Predictor')
st.markdown("**Calculate your mental Health Score**")

left_column, right_column = st.columns([1.2, 0.8], gap="large")
with left_column:
    st.subheader("Student Form", text_alignment="left")

form_area = left_column.container(height=520, border=False)
with form_area.form(key="registration_form"):
    st.write("Please fill out your details:")
    
    # 2. Add input widgets inside the block
    name = st.text_input(
        "Full Name", placeholder="Enter your full name", required=True
    )
    age = st.number_input(
        "Age",
        placeholder="Enter your age",
        required=True,
        step=1,
        value=None,
        min_value=1,
    )
    Gender = st.selectbox(
        "Gender",
        ["Male", "Female"],
        placeholder="Select your gender",
        index=None,
        required=True,
    )
    country = st.selectbox(
        "Country",
        [
            "Afghanistan", "Albania", "Andorra", "Argentina", "Armenia", "Australia",
            "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Belarus",
            "Belgium", "Bhutan", "Bolivia", "Bosnia", "Brazil", "Bulgaria", "Canada",
            "Chile", "China", "Colombia", "Costa Rica", "Croatia", "Cyprus",
            "Czech Republic", "Denmark", "Ecuador", "Egypt", "Estonia", "Finland",
            "France", "Georgia", "Germany", "Ghana", "Greece", "Hong Kong", "Hungary",
            "Iceland", "India", "Indonesia", "Iraq", "Ireland", "Israel", "Italy",
            "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kosovo", "Kuwait",
            "Kyrgyzstan", "Latvia", "Lebanon", "Liechtenstein", "Lithuania",
            "Luxembourg", "Malaysia", "Maldives", "Malta", "Mexico", "Moldova",
            "Monaco", "Montenegro", "Morocco", "Nepal", "Netherlands", "New Zealand",
            "Nigeria", "North Macedonia", "Norway", "Oman", "Other", "Pakistan",
            "Panama", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar",
            "Romania", "Russia", "San Marino", "Serbia", "Singapore", "Slovakia",
            "Slovenia", "South Africa", "South Korea", "Spain", "Sri Lanka", "Sweden",
            "Switzerland", "Syria", "Taiwan", "Tajikistan", "Thailand", "Trinidad",
            "Turkey", "UAE", "UK", "USA", "Ukraine", "Uruguay", "Uzbekistan",
            "Vatican City", "Venezuela", "Vietnam", "Yemen",
        ],
        placeholder="Select your country",
        index=None,
        required=True,
    )
    academic_level = st.selectbox(
        "Academic Level",
        ["Graduate", "High School", "Undergraduate"],
        placeholder="Select your academic level",
        index=None,
        required=True,
    )
    most_used_platform = st.selectbox(
        "Most Used Platform",
        [
            "Facebook", "Instagram", "KakaoTalk", "LINE", "LinkedIn", "Snapchat",
            "TikTok", "Twitter", "VKontakte", "WeChat", "WhatsApp", "YouTube",
        ],
        placeholder="Select your most-used platform",
        index=None,
        required=True,
    )
    purpose_of_use = st.selectbox(
        "Purpose of Use",
        ["Education", "Entertainment", "Networking", "News"],
        placeholder="Select your purpose of use",
        index=None,
        required=True,
    )
    avg_daily_usage_hours = st.number_input(
        "Average Daily Usage (hours)",
        min_value=0.0,
        max_value=24.0,
        value=None,
        step=0.5,
        placeholder="Enter hours from 0 to 24",
        required=True,
    )
    daily_unlocks = st.number_input(
        "Daily Unlocks",
        min_value=0,
        value=None,
        step=1,
        placeholder="Enter the number of daily unlocks",
        required=True,
    )
    study_hours = st.number_input(
        "Study Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=None,
        step=0.5,
        placeholder="Enter hours from 0 to 24",
        required=True,
    )
    physical_activity_hours = st.number_input(
        "Physical Activity Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=None,
        step=0.5,
        placeholder="Enter hours from 0 to 24",
        required=True,
    )
    sleep_hours = st.number_input(
        "Sleep Hours per Night",
        min_value=0.0,
        max_value=24.0,
        value=None,
        step=0.5,
        placeholder="Enter hours from 0 to 24",
        required=True,
    )
    stress_level = st.selectbox(
        "Stress Level",
        ["Low", "Medium", "High", "Very High"],
        placeholder="Select your stress level",
        index=None,
        required=True,
    )
    confirmed = st.checkbox("All the required things are correct")
    
    # 3. Every form must have a submit button
    submitted = st.form_submit_button("Submit")

score = None
error_message = None

if submitted:
    required_values = (
        name,
        age,
        Gender,
        country,
        academic_level,
        most_used_platform,
        purpose_of_use,
        avg_daily_usage_hours,
        daily_unlocks,
        study_hours,
        physical_activity_hours,
        sleep_hours,
        stress_level,
    )
    if any(value is None or value == "" for value in required_values):
        error_message = "Please complete all required fields."
    elif not confirmed:
        error_message = "Please confirm that all required details are correct."
    else:
        payload = {
            "Age": int(age),
            "Gender": Gender,
            "Country": country,
            "Academic_level": academic_level,
            "Most_Used_Platform": most_used_platform,
            "Purpose_of_Use": purpose_of_use,
            "Avg_Daily_Usage_Hours": avg_daily_usage_hours,
            "Daily_Unlocks": int(daily_unlocks),
            "Study_Hours": study_hours,
            "Physical_Activity_Hours": physical_activity_hours,
            "Sleep_Hours_Per_Night": sleep_hours,
            "Stress_Level": stress_level,
        }
        try:
            score = get_prediction(payload)
        except HTTPError as error:
            error_message = f"The prediction API returned an HTTP {error.code} error."
        except URLError:
            error_message = (
                "Could not connect to the prediction API. Start it with "
                "`uvicorn main:app --reload`."
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            error_message = f"Invalid prediction API response: {error}"

with right_column:
    st.subheader("Prediction")
    if error_message:
        st.error(error_message)
    elif score is not None:
        st.success(
            f"Thank you, {name}! Your predicted mental health score is {score:.2f}."
        )
        st.subheader("Predicted Score")
        render_score_gauge(score)
    else:
        st.info("Complete the form and submit to see your predicted score here.")
