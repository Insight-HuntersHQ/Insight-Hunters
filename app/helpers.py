import streamlit as st
import plotly.express as px
import pandas as pd
from pathlib import Path
from PIL import Image
from tb_model_inference import (
    load_tb_model,
    load_validator_model,
    run_complete_inference,
)


def global_style():
    st.markdown(
        """<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" />
<style>
.theme-hover-tile {
    background-color: #10253A;
    padding: 24px;
    border-radius: 12px;
    border: 1px solid #10253A;
    color: #EDF7FF;
    min-height: 210px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}
.theme-hover-tile:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(100, 232, 224, 0.15);
    border-color: #6495ED;
}
.tile-title {
    color: #6495ED;
    margin: 0 0 8px 0;
    font-weight: 600;
    font-size: 22px;
    line-height: 1.3;
    display: flex;
    align-items: center;
    gap: 8px;
    overflow-wrap: normal;
    word-break: normal;
}
.tile-title .material-symbols-outlined {
    flex-shrink: 0;
}
.material-symbols-outlined {
    font-size: 24px;
}
</style>""",
        unsafe_allow_html=True,
    )


def tile(title, text, icon=None):
    icon_html = f'<span class="material-symbols-outlined">{icon}</span>' if icon else ""
    return st.markdown(
        f"""
        <div class="theme-hover-tile">
            <div>
                <h3 class="tile-title">{icon_html} {title}</h3>
                <p style="color: #EDF7FF; margin: 0; font-size: 14px; line-height: 1.4; opacity: 0.85;">{text}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def tb_cat_style():
    st.markdown(
        """
        <style>
        .dash-card {
            background-color: #10253A;
            border: 1px solid rgba(100, 149, 237, 0.2);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
        }
        .banner-positive {
            background-color: rgba(255, 75, 75, 0.12);
            border: 1px solid rgba(255, 75, 75, 0.4);
            border-radius: 10px;
            padding: 16px;
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .banner-negative {
            background-color: rgba(100, 232, 224, 0.12);
            border: 1px solid rgba(100, 232, 224, 0.4);
            border-radius: 10px;
            padding: 16px;
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .colorbar-legend {
            width: 14px;
            height: 180px;
            background: linear-gradient(to top, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000);
            border-radius: 4px;
        }
        [data-testid="stFileUploader"] {
            width: 100%;
        }
        [data-testid="stFileUploader"] section {
            padding: 8px 12px !important;
            min-height: unset !important;
            background-color: #10253A !important;
            border: 1px dashed rgba(100, 149, 237, 0.4) !important;
            border-radius: 8px !important;
        }

        [data-testid="stFileUploader"] section svg {
            display: none !important;
        }

        [data-testid="stFileUploader"] section span {
            font-size: 13px !important;
            color: #6495ED !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main_page():
    global_style()
    st.markdown(
        "<h1 style='color:cornflowerblue;'>🫁 Insight Hunters</h1>"
        "<h2 style='color:white;'>Tuberculosis Detection Dashboard</h2>",
        unsafe_allow_html=True
    )
    st.write(
        "**Deep learning-driven analysis of chest X-ray images to allow for earlier detection of Tuberculosis (TB) in resource-constrained settings.**")
    st.divider()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        tile("Early Detection", "Accurate and efficient diagnosis support for better identification of Tuberculosis.",
             icon="fact_check")

    with col2:
        tile("Explainable AI", "Model's decisions made understandable and are visualised using Grad-CAM.", icon="chat")

    with col3:
        tile("Real-World Impact", "Designed with real-world applications in mind.", icon="groups")

    with col4:
        tile("Data-Driven Decisions",
             "Provides healthcare professionals with reliable, transparent & actionable insights.", icon="analytics")

    st.divider()

    col5, col6 = st.columns([1, 1.5])

    with col5:
        tile("Addressing the TB Challenge",
             "Tuberculosis remains one of the deadliest infectious diseases accross the world. In resource-constrained environments, access to radiologists is often limited. This model aims to bridge that gap by providing medical professionals with automated, efficient screening support.",
             icon="public")

    with col6:
        hb_data = {
            "country_name": [
                "Bangladesh", "Brazil", "China", "Democratic Republic of Congo ", "Ethiopia",
                "India", "Indonesia", "Kenya", "Mozambique", "Myanmar",
                "Nigeria", "Pakistan", "Philippines", "South Africa", "Thailand",
                "Vietnam", "Angola", "Cambodia", "Central African Republic", "Congo",
                "Gabon", "Democratic People's Republic of Korea", "Lesotho", "Liberia", "Namibia",
                "Papua New Guinea", "Sierra Leone", "Tanzania", "Uganda", "Zambia"
            ],
            # ISO codes for plotly maps
            "iso_code": [
                "BGD", "BRA", "CHN", "COD", "ETH",
                "IND", "IDN", "KEN", "MOZ", "MMR",
                "NGA", "PAK", "PHL", "ZAF", "THA",
                "VNM", "AGO", "KHM", "CAF", "COG",
                "GAB", "PRK", "LSO", "LBR", "NAM",
                "PNG", "SLE", "TZA", "UGA", "ZMB"
            ]
        }
        df_hb = pd.DataFrame(hb_data)
        df_hb["burden_level"] = "High Burden"

        fig = px.choropleth(df_hb, locations="iso_code", color="burden_level",
                            color_discrete_map={"High Burden": "#4682B4"},
                            hover_name="country_name", projection="natural earth",
                            title="High TB Burdened Countries According To WHO"
                            )

        fig.update_layout(
            title=dict(
                font=dict(color="#6495ED", size=18, family="Montserrat"),
                x=0.00,
                y=0.92,
                xanchor="left",
                yanchor="bottom"
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin={"r": 0, "t": 35, "l": 0, "b": 0},
            title_font_color="#6495ED",
            title_font_size=18,
            title_x=0.05,
            showlegend=False,
            height=260,
            geo=dict(
                bgcolor="rgba(0,0,0,0)",
                lakecolor="rgba(0,0,0,0)",
                rivercolor="rgba(0,0,0,0)",
                showcountries=True,
                countrycolor="#1e3a5f",
                showcoastlines=True,
                coastlinecolor="#1e3a5f"
            ),
        )
        fig.update_traces(
            marker=dict(line=dict(width=1.2, color="#6495ED")),
            colorbar=dict(bgcolor="rgba(0,0,0,0)"),
            hovertemplate="<b>%{hovertext}</b><br>High TB Burdened Countries According To WHO"
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def tb_cat_page():
    global_style()
    tb_cat_style()

    st.title(
        "Tuberculosis Detection via Deep Learning",
        anchor=False,
    )
    st.write(
        "Upload a patient's chest X-ray image to get AI-powered "
        "insights related to their TB status."
    )

    col_left, col_right = st.columns([1.1, 1.3], gap="large")

    MODEL_DIR = Path(__file__).resolve().parent

    @st.cache_resource
    def get_models():
        tb_model = load_tb_model(
            MODEL_DIR / "nih_tb_final_selected_compat.keras"
        )

        validator_model = load_validator_model(
            MODEL_DIR / "mobilenetv2_validator_best_compact.keras"
        )

        return tb_model, validator_model

    result = None
    uploaded_file = None

    with col_left:
        st.markdown(
            """
            <div style="font-weight:600; color:#EDF7FF;
                        margin-bottom:8px; font-size:15px;">
                Uploaded Scan
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload X-ray",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:

                image = Image.open(uploaded_file)
                image_size = image.size
                image_format = image.format or "Unknown"

                st.markdown(
                    f"""
                    <div style="display:flex; align-items:center;
                                justify-content:space-between;
                                margin-bottom:12px; font-size:13px;
                                color:#8da4c4;" class="tech-text">
                        <span style="color:#EDF7FF;">
                            <span class="material-symbols-outlined"
                                  style="font-size:18px;
                                         color:#6495ED;
                                         vertical-align:middle;">
                                draft
                            </span>
                            {uploaded_file.name}
                        </span>
                        <span>{image_format}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.image(image, use_container_width=True)

                st.markdown(
                    f"""
                    <div style="margin-top:12px; display:flex;
                                justify-content:space-between;
                                font-size:12px; color:#8da4c4;"
                         class="tech-text">
                        <span>
                            Image size: {image_size[0]} × {image_size[1]}
                        </span>
                        <span>Format: {image_format}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Load the model only when an image is uploaded
                with st.spinner("Analysing X-ray..."):
                    tb_model, validator_model = get_models()
                    uploaded_file.seek(0)
                    result = run_complete_inference(
                        tb_model,
                        validator_model,
                        uploaded_file,
                    )

            except Exception as exc:
                st.error(
                    "The image could not be analysed. Check that the "
                    "model file is available and compatible with your "
                    "TensorFlow/Keras environment, then try again."
                )
                st.exception(exc)

        else:
            st.info(
                "Upload a chest X-ray file to initiate scan analysis."
            )

    # results and gradcam col
    with col_right:
        if result is not None:
            status = result.get("status")

            if status == "error":
                st.error(result.get("message", "An error occurred during analysis."))

            elif status == "rejected":
                st.warning(result.get("message", "This image was rejected by the validator."))

            elif status == "accepted":
                prediction = result["predicted_class"]
                tb_probability = float(result["tb_probability"])
                is_positive = prediction == "TB Positive"

                result_color = "#ff4b4b" if is_positive else "#64E8E0"
                result_icon = "error" if is_positive else "check_circle"
                banner_class = (
                    "banner-positive" if is_positive else "banner-negative"
                )
                result_message = (
                    "The model detected patterns associated with tuberculosis."
                    if is_positive
                    else "The model did not detect patterns associated "
                         "with tuberculosis."
                )
                score_pct = (
                    tb_probability * 100
                    if is_positive
                    else (1 - tb_probability) * 100
                )

                st.markdown(
                    f"""
<div style="font-weight:600; color:#EDF7FF; margin-bottom:8px; font-size:15px;">Classification Result</div>
<div class="dash-card">
<div class="{banner_class}" style="display:flex; align-items:center; gap:14px;">
<span class="material-symbols-outlined" style="font-size:36px; color:{result_color};">{result_icon}</span>
<div>
<div style="color:{result_color}; font-weight:700; font-size:18px;">{prediction}</div>
<div style="color:#EDF7FF; font-size:13px; opacity:0.85;">{result_message}</div>
</div>
</div>
</div>
<div class="dash-card">
<div style="font-size:13px; color:#8da4c4; margin-bottom:10px;">Predicted-class score</div>
<div style="display:flex; align-items:center; gap:16px;">
<div style="background:rgba(100,232,224,0.1); border:3px solid #64E8E0; border-radius:50%; width:64px; height:64px; display:flex; align-items:center; justify-content:center; font-weight:bold; color:#64E8E0; font-size:14px; flex-shrink:0;">{score_pct:.1f}%</div>
<div>
<div style="color:#64E8E0; font-weight:600; font-size:14px;">Model output</div>
<div style="color:#EDF7FF; font-size:12px; opacity:0.8;">Score for the predicted class: {score_pct:.1f}%.</div>
</div>
</div>
</div>
""",
                    unsafe_allow_html=True,
                )

                # Grad-CAM visualization
                st.markdown(
                    """
                    <div style="font-size:13px; color:#EDF7FF;
                                font-weight:600; margin:16px 0 12px;">
                        X-ray Comparison
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if result.get("overlay") is not None:
                    image_col, gradcam_col = st.columns(2, gap="medium")

                    with image_col:
                        st.markdown(
                            """
                            <div style="font-size:12px; color:#8da4c4;
                                        margin-bottom:8px;">
                                Original X-ray
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.image(
                            image,
                            use_container_width=True,
                        )

                    with gradcam_col:
                        st.markdown(
                            """
                            <div style="font-size:12px; color:#8da4c4;
                                        margin-bottom:8px;">
                                Grad-CAM Overlay
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.image(
                            result["overlay"],
                            use_container_width=True,
                        )
                else:
                    st.info(
                        "Grad-CAM visualization is not available."
                    )

                st.markdown(
                    """
                    <div class="dash-card" style="padding:14px 20px;">
                        <div style="display:flex; gap:10px;
                                    align-items:flex-start;">
                            <span class="material-symbols-outlined"
                                  style="font-size:18px; color:#6495ED;
                                         margin-top:1px;">
                                info
                            </span>
                            <div>
                                <div style="font-size:12px;
                                            font-weight:600;
                                            color:#EDF7FF;
                                            margin-bottom:2px;">
                                    What is Grad-CAM?
                                </div>
                                <div style="font-size:11px;
                                            color:#8da4c4;
                                            line-height:1.4;">
                                    Grad-CAM highlights image regions
                                    that contributed to the model's
                                    prediction. It is an explanatory
                                    visualization, not proof that ther
                                    model focused on clinically relevant
                                    findings.
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:
                st.markdown(
                    """
                    <div class="dash-card">
                        <div style="font-size:15px; font-weight:600;
                                    color:#EDF7FF; margin-bottom:8px;">
                            Classification Result
                        </div>
                        <div style="font-size:13px; color:#8da4c4;">
                            Your results will appear here after you
                            upload a chest X-ray.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.error(
        "Disclaimer: This model is developed soley for academic research purposes. Any output should not be used as clincial advice. Final clincal diagnosis and treatment-related decisions should be made by a recognized healthcare profession",
        icon=":material/warning:")


def info_page():
    st.markdown("<h1 style='color:cornflowerblue;'>Information</h1>", unsafe_allow_html=True)
    st.write("")
    st.subheader("🦠 Information on Tuberculosis")

    tab1, tab2, tab3, tab4 = st.tabs([
        ":material/assignment: Overview & Symptoms",
        ":material/social_distance: Transmission",
        ":material/clinical_notes: Diagnosis & Testing",
        ":material/medication: Treatment & Care"
    ])

    with tab1:
        st.markdown("#### What is Tuberculosis?")
        st.write(
            "**Tuberculosis (TB)** is a severe bacterial infection that primarily affects the **lungs**. However, it can also affect"
            "the spine, brain, kidneys or other various parts of your body."
        )

        st.markdown("##### Primary Symptoms to aware of:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(":material/air: **A long-lasting cough**")
            st.markdown(":material/water_drop: **Coughing up blood** or mucus")
            st.markdown(":material/personal_injury: **Chest pain** or feeling pain when breathing")
        with col2:
            st.markdown(":material/monitor_weight_loss: **Weight** and appetite loss")
            st.markdown(":material/sick: **Sweating at night** and fever")
            st.markdown(":material/hotel: **Fatigue** or intense weakness")

    with tab2:
        st.markdown("#### How TB Spreads")
        st.info(":material/masks: **Airborne Transmission:** TB is spread through the air from one person to another.")
        st.markdown(
            "You can contact TB by breathing in the bacteria when someone infected coughs, sneezes or talks in close proximity to you."
            "Usually you need regular, close contact with an infected person."
        )

        with st.expander(":material/help: Difference between Latent TB Infection and Active TB Disease"):
            st.markdown(""" A **Latent infection** means that you are infected but do not get sick from the infection and cannot transmit it to others. Whereas, an **Active
            infection** means that you experience symptoms and are contagious.  
            """)

    with tab3:
        st.markdown("#### Testing and Diagnosis")
        st.write("Healthcare providers generally employ two tests for the diagnosis of TB:")

        st.markdown("1. **:material/radiology: Imaging:** Imaging such as X-rays and CT scans are used to diagnose TB.")
        st.markdown("2. **:material/science: Sputum(mucus):** This is the best way to diagnose active TB.")

    with tab4:
        st.markdown("#### Treatment & Medical Management")
        st.warning(
            ":material/warning: **Important:** TB is curable, but if you do not complete the full course of treatment, TB can return and be harder to cure.")

        st.markdown("### :material/pill: Common Treatment Protocol:")
        st.markdown(
            "Infected patients, both active and passive, are usually prescribe a treatment course of multiple anti-biotics.")

        st.markdown("---")
        st.caption(
            "Disclaimer: This application is for educational purposes only. Always consult a qualified healthcare professional for medical advice, diagnoses, or treatment.")

    st.caption(
        "Source: [Cleveland Clinic](https://my.clevelandclinic.org/health/diseases/11301-tuberculosis) (Updated Aug 2026)")

    st.divider()

    st.markdown(
        "<h3 style='color:orange;'>🧠 Model Information</h3>",
        unsafe_allow_html=True
    )
    st.markdown("##### Selected model: `ResNet50`")
    st.markdown("###### Comparison of all models: ")
    metric_df = pd.read_csv("four_model_test_comparison.csv")
    st.table(metric_df)
    st.caption("ResNet50 selected based on better overall metrics compared to the other three models")

    st.error(
        "Disclaimer: This model is developed soley for academic research purposes. Any output should not be used as clincial advice. Final clincal diagnosis and treatment-related decisions should be made by a recognized healthcare profession",
        icon=":material/warning:")
