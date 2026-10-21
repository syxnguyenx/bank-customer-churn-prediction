from pathlib import Path

import numpy as np
import pandas as pd
import shap
import streamlit as st

from inference import (
    INPUT_COLUMNS,
    THRESHOLD,
    encode_customers,
    load_artifacts as load_inference_artifacts,
    predict_customers,
    risk_level,
)


APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
)
st.title("Dự đoán khách hàng rời bỏ ngân hàng")
st.write("Nhập thông tin khách hàng hoặc tải CSV để dự đoán nguy cơ rời bỏ bằng mô hình XGBoost đã huấn luyện.")


@st.cache_resource
def load_artifacts():
    return load_inference_artifacts(APP_DIR)


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model, model_output="raw")


def get_positive_class_shap(shap_result, feature_count):
    values = np.asarray(getattr(shap_result, "values", shap_result))
    if values.ndim == 2 and values.shape == (1, feature_count):
        return values[0]
    if values.ndim == 3:
        if values.shape[0] == 1 and values.shape[1] == feature_count:
            return values[0, :, 1] if values.shape[2] > 1 else values[0, :, 0]
        if values.shape[1] == 1 and values.shape[2] == feature_count:
            return values[1, 0, :] if values.shape[0] > 1 else values[0, 0, :]
    raise ValueError(f"Kích thước SHAP không khớp với {feature_count} feature: {values.shape}")


try:
    model, feature_columns = load_artifacts()
except Exception as exc:
    st.error("Không tải được model hoặc danh sách feature. Kiểm tra các artifact cạnh app.py.")
    st.exception(exc)
    st.stop()


st.subheader("Thông tin khách hàng")
with st.form("customer_form"):
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=1)
        geography = st.selectbox("Quốc gia (Geography)", ["France", "Germany", "Spain"])
        gender = st.selectbox("Giới tính (Gender)", ["Female", "Male"])
        age = st.number_input("Tuổi (Age)", min_value=18, max_value=100, value=40, step=1)
        tenure = st.number_input("Số năm sử dụng dịch vụ (Tenure)", min_value=0, max_value=10, value=5, step=1)
    with col2:
        balance = st.number_input("Số dư tài khoản (Balance)", min_value=0.0, value=75000.0, step=1000.0)
        num_products = st.number_input("Số sản phẩm sử dụng (NumOfProducts)", min_value=1, max_value=4, value=2, step=1)
        has_card = st.selectbox("Có thẻ tín dụng (HasCrCard)", options=[1, 0], format_func=lambda value: "Có" if value == 1 else "Không")
        active_member = st.selectbox("Là thành viên hoạt động (IsActiveMember)", options=[1, 0], format_func=lambda value: "Có" if value == 1 else "Không")
        estimated_salary = st.number_input("Lương ước tính (EstimatedSalary)", min_value=0.0, value=100000.0, step=1000.0)
    submitted = st.form_submit_button("Dự đoán nguy cơ rời bỏ", use_container_width=True)

if submitted:
    form_data = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_card,
        "IsActiveMember": active_member,
        "EstimatedSalary": estimated_salary,
    }])
    try:
        customer_features = encode_customers(form_data, feature_columns)
        probability = float(predict_customers(customer_features, model)[0])
        predicted_churn = probability >= THRESHOLD

        st.divider()
        st.subheader("Kết quả dự đoán")
        result_col1, result_col2, result_col3 = st.columns(3)
        result_col1.metric("Xác suất rời bỏ", f"{probability:.2%}")
        result_col2.metric("Mức rủi ro", risk_level(probability))
        result_col3.metric("Ngưỡng phân loại", f"{THRESHOLD:.0%}")
        if predicted_churn:
            st.warning(f"Mô hình xếp khách hàng vào nhóm có nguy cơ rời bỏ theo ngưỡng {THRESHOLD:.0%}.")
        else:
            st.success(f"Mô hình chưa xếp khách hàng vào nhóm có nguy cơ rời bỏ theo ngưỡng {THRESHOLD:.0%}.")
        st.caption("Đây là xác suất dự đoán của model, không phải sự chắc chắn rằng khách hàng sẽ rời bỏ.")

        st.divider()
        st.subheader("Giải thích dự đoán bằng SHAP")
        try:
            explainer = load_explainer(model)
            shap_result = explainer(customer_features)
            local_values = get_positive_class_shap(shap_result, len(feature_columns))
            waterfall_figure = None
            try:
                import matplotlib.pyplot as plt

                waterfall_figure = plt.figure()
                shap.plots.waterfall(
                    shap_result[0], max_display=len(feature_columns), show=False
                )
                st.pyplot(waterfall_figure)
                st.caption("Waterfall thể hiện các đóng góp từ giá trị nền E[f(X)] đến đầu ra f(x) của khách hàng, theo đơn vị raw; không phải phần trăm xác suất.")
            except Exception as exc:
                st.warning("Không thể hiển thị biểu đồ SHAP Waterfall. Kết quả dự đoán vẫn ở phía trên; bảng SHAP hiển thị bên dưới.")
                st.exception(exc)
            finally:
                if waterfall_figure is not None:
                    plt.close(waterfall_figure)

            explanation = pd.DataFrame({
                "Đặc trưng": customer_features.columns,
                "Giá trị đầu vào": customer_features.iloc[0].values,
                "SHAP impact (raw)": local_values,
            })
            explanation["Hướng tác động"] = np.where(
                explanation["SHAP impact (raw)"] >= 0,
                "Tăng xu hướng rời bỏ",
                "Giảm xu hướng rời bỏ",
            )
            explanation["Độ lớn tác động"] = explanation["SHAP impact (raw)"].abs()
            st.write("Các đặc trưng có ảnh hưởng lớn nhất đến dự đoán:")
            st.dataframe(
                explanation.sort_values("Độ lớn tác động", ascending=False)
                .head(8)[["Đặc trưng", "Giá trị đầu vào", "SHAP impact (raw)", "Hướng tác động"]],
                use_container_width=True,
                hide_index=True,
            )
            st.caption("SHAP được tính trên đầu ra raw của XGBoost (thường là log-odds), không phải điểm phần trăm xác suất. Dấu dương đẩy đầu ra về phía churn; dấu âm đẩy về phía ở lại.")
        except Exception as exc:
            st.warning("Dự đoán đã hoàn thành nhưng phần giải thích SHAP không chạy được.")
            st.exception(exc)
    except Exception as exc:
        st.error("Không thể dự đoán khách hàng này.")
        st.exception(exc)


st.divider()
st.subheader("Dự đoán hàng loạt từ CSV")
st.write("CSV cần có các cột: " + ", ".join(INPUT_COLUMNS) + ". Geography nhận France/Germany/Spain; Gender nhận Female/Male.")
uploaded_file = st.file_uploader("Tải file CSV khách hàng", type=["csv"], key="batch_csv")
if uploaded_file is not None:
    try:
        batch_input = pd.read_csv(uploaded_file)
        if batch_input.empty:
            st.error("File CSV không có dòng dữ liệu.")
        else:
            batch_features = encode_customers(batch_input, feature_columns)
            batch_probabilities = predict_customers(batch_features, model)
            batch_results = batch_input.copy()
            batch_results["ChurnProbability"] = batch_probabilities
            batch_results["PredictedChurn"] = (batch_probabilities >= THRESHOLD).astype(int)
            batch_results["RiskLevel"] = [risk_level(value) for value in batch_probabilities]
            st.success(f"Đã dự đoán {len(batch_results):,} khách hàng với ngưỡng {THRESHOLD:.0%}.")
            st.dataframe(batch_results, use_container_width=True, hide_index=True)
            st.download_button(
                "Tải kết quả CSV",
                data=batch_results.to_csv(index=False).encode("utf-8-sig"),
                file_name="churn_predictions.csv",
                mime="text/csv",
            )
    except Exception as exc:
        st.error("Không thể xử lý CSV. Kiểm tra tên cột, giá trị và định dạng file.")
        st.exception(exc)
