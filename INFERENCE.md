# Suy luận độc lập và tích hợp backend

`inference.py` là nguồn logic dùng chung cho Streamlit và Python/backend. Import
module không khởi chạy Streamlit, không nạp model và không yêu cầu SHAP. Model
chỉ được nạp khi gọi dự đoán hoặc `load_artifacts()` / `load_bundle()`.

## Cài đặt dependency

Môi trường đã được người dùng kiểm thử dùng Python 3.12.9. Trên máy mới, vào
thư mục chứa project, tạo `.venv` nếu chưa có rồi cài dependency:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Nếu đã có môi trường phù hợp, chỉ chạy lệnh cài từ requirements bằng interpreter
của môi trường đó. `requirements.txt` ghim tám dependency trực tiếp theo metadata
cài đặt thực tế trong `.venv` hiện tại, đủ cho app, inference và các unittest.
`unittest` không cần cài thêm. File này không khóa toàn bộ dependency gián tiếp
và chưa được xác nhận bằng một lần cài mới trên mọi nền tảng. FastAPI/Uvicorn
chỉ cài thêm nếu triển khai adapter API tùy chọn.

## File triển khai

- `inference.py`: validation, encoding, prediction và threshold.
- `requirements.txt`: dependency trực tiếp được ghim để cài trên môi trường đích.
- `.gitattributes`: giữ LF cho Python và byte binary của artifact khi checkout.
- `inference_bundle_portable.joblib`: bundle portable định dạng v2, nhúng nguyên
  byte của `best_xgb.joblib` và `feature_columns.joblib`.
- Hai artifact gốc vẫn ở project để app sử dụng trực tiếp; không cần chép chúng
  riêng khi triển khai bằng bundle portable.

Mặc định `build_bundle()` và CLI dùng `mode="portable"`. Khi triển khai sang máy
hoặc ổ đĩa khác, chỉ cần module `inference.py` và bundle portable, cùng các
dependency tương thích. Bundle không chứa đường dẫn đến máy nguồn. Loader kiểm
tra checksum của cả hai phần dữ liệu nhúng trước khi nạp chúng qua bộ nhớ.
Đây là sao chép byte gốc vào gói, không nạp rồi serialize lại XGBoost.

Bundle giữ SHA-256, schema, 11 feature theo thứ tự, threshold `0.28`, nhãn dương
`1`, phiên bản định dạng và metadata đóng gói. SHA-256 của `inference.py` giúp
phát hiện module không khớp bundle. Chỉ nạp các file joblib đáng tin cậy. Khi
thay module, tạo bundle tên mới; không sửa bundle cũ. Bundle v1 trước thay đổi
này không khớp checksum module mới: hãy tạo bundle v2, không bỏ kiểm tra checksum.

Chế độ tùy chọn `mode="reference"` / `--mode reference` không nhúng artifact:
đường dẫn cùng ổ đĩa được lưu tương đối với thư mục bundle; khác ổ đĩa hoặc UNC
share thì lưu tuyệt đối và đánh dấu `path_type="absolute"`. Loại này cần hai
file ngoài bundle. Tham chiếu tương đối chỉ di chuyển được nếu giữ cấu trúc thư
mục; tham chiếu tuyệt đối phụ thuộc đường dẫn máy nguồn, nên cần tạo bundle mới
ở đích hoặc chọn portable. Loader không đoán hay âm thầm tìm artifact ở nơi khác.

Không thêm Scaler. Notebook huấn luyện chưa có nên việc dùng Scaler lúc train
chưa được xác minh. Phiên bản package trong metadata là môi trường đóng gói,
phiên bản trong requirements là môi trường hiện tại; chúng không phải phiên bản
được xác nhận lúc huấn luyện. Tạo bundle không đồng nghĩa
đã kiểm thử runtime hoặc đã xác nhận tương thích phiên bản XGBoost.

## Gọi từ Python

Chạy từ `D:\Project`, hoặc cài/đặt module vào đường dẫn import của backend:

```python
from inference import predict_customer

customer = {
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Female",
    "Age": 40,
    "Tenure": 5,
    "Balance": 75000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 100000.0,
}

result = predict_customer(customer, bundle_path="inference_bundle_portable.joblib")
print(result)
# {
#   "churn_probability": <float từ model>,
#   "predicted_churn": 0 hoặc 1,
#   "risk_level": "Thấp" / "Trung bình" / "Cao",
#   "threshold": 0.28,
# }
```

`predict_customer(customer)` cũng dùng trực tiếp hai artifact cạnh `inference.py`.
Để dùng thư mục khác, truyền `artifact_dir=...`; không truyền đồng thời
`artifact_dir` và `bundle_path`. Backend nên dùng `bundle_path` tuyệt đối để
không phụ thuộc thư mục làm việc. Kết quả chỉ chứa kiểu Python tương thích JSON.
Model được cache theo đường dẫn trong từng process; khởi động lại backend khi
triển khai release mới. Lần nạp bundle đầu tiên kiểm tra schema và checksum.

Không điền giá trị mặc định cho payload thiếu trường: validation vẫn từ chối như
app cũ. `predicted_churn = int(churn_probability >= 0.28)`. Mức rủi ro vẫn là
Thấp khi `< 0.30`, Trung bình khi `< 0.60`, còn lại là Cao; do đó khoảng
`[0.28, 0.30)` có nhãn churn `1` nhưng mức rủi ro Thấp, giống ứng dụng trước đó.

## Ví dụ FastAPI (tùy chọn, chưa tạo server trong project)

Nếu backend đã dùng FastAPI, có thể đưa phần sau vào module API của backend:

```python
from pathlib import Path
from fastapi import FastAPI, HTTPException
from inference import predict_customer

api = FastAPI()
BUNDLE = Path(__file__).resolve().parent / "inference_bundle_portable.joblib"

@api.post("/predict")
def predict(payload: dict):
    try:
        return predict_customer(payload, bundle_path=BUNDLE)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
```

Ví dụ này là hướng dẫn adapter; FastAPI không phải dependency của module suy
luận. Endpoint nhận JSON có 10 trường như ví dụ Python. Với production, nạp và
kiểm tra bundle lúc startup để lỗi artifact/deployment được tách khỏi lỗi input.

## Batch và SHAP

```python
import pandas as pd
from inference import load_bundle, encode_customers, predict_customers, THRESHOLD

model, columns = load_bundle("inference_bundle_portable.joblib")
raw = pd.read_csv("customers.csv")
features = encode_customers(raw, columns)
probabilities = predict_customers(features, model)
labels = (probabilities >= THRESHOLD).astype(int)
```

Streamlit dùng chính các hàm này với artifact gốc. `customer_features` và `model`
vẫn được giữ để tính SHAP Waterfall theo raw output như trước; module backend
không thêm SHAP vào đường dự đoán.

## Kiểm thử và đóng gói trong PowerShell

```powershell
Set-Location D:\Project
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Test so sánh logic mới với fixture đóng băng từ app trước khi tách module:
DataFrame và xác suất phải giống nhau chính xác, thông báo lỗi giữ nguyên,
threshold/mức cảnh báo giữ nguyên. Có kiểm tra input mặc định, tổ hợp quốc
gia/giới tính, biên hợp lệ, dữ liệu lỗi, batch, JSON, import độc lập và bundle.
Fixture `tests/fixtures/legacy_app_core.py` chỉ là chuẩn đối chiếu cho test;
không được app hoặc backend sử dụng.

Test bundle còn kiểm tra lỗi khác ổ Windows, fallback của reference, bundle
portable không gọi `relpath()`, chuyển bundle sang thư mục mới mà không đọc hai
artifact nguồn, dữ liệu nhúng bị sửa, schema/feature order/threshold và checksum.
Các test so sánh dự đoán dùng model thật, không thay bằng model giả để bỏ qua lỗi.

Nếu cần tạo release bundle từ môi trường Python của bạn, chọn tên mới:

```powershell
& .\.venv\Scripts\python.exe package_inference.py --mode portable --output inference_bundle_portable_local.joblib
```

Lệnh từ chối ghi đè file đã tồn tại và không nạp/fit/lưu lại XGBoost. Với portable,
bundle ở ổ khác artifact vẫn tạo và nạp được; không cần `os.path.relpath()`.
Giữ nguyên byte của `inference.py` để khớp checksum. Nếu chủ động cần gói tham chiếu:

```powershell
& .\.venv\Scripts\python.exe package_inference.py --mode reference --output inference_bundle_reference_local.joblib
```

Chỉ phân phối `best_xgb.joblib`, `feature_columns.joblib` và
`inference_bundle_portable.joblib` hiện hành. Bundle `inference_bundle.joblib`
v1 cũ là file lưu trữ cục bộ bị `.gitignore` loại trừ, không thuộc bộ bàn giao.
`build_bundle()` không truyền `output` vẫn chọn tên `inference_bundle.joblib`
và từ chối nếu file đích đã tồn tại; luôn truyền tên mới như các lệnh trên.

Giữ `.gitattributes` trong repository để file Python được checkout với LF.
Không đổi BOM hoặc kiểu xuống dòng của `inference.py`: bundle kiểm tra checksum
theo byte của module. Tài liệu, requirements và cấu hình Git có thể cập nhật mà
không cần tạo lại bundle nếu module và artifact không thay đổi.

Kiểm tra UI sau khi tách: reload Streamlit, dự đoán bản ghi mặc định, đổi một
khách hàng, kiểm tra Waterfall và bảng SHAP, rồi upload CSV mẫu và tải kết quả.
Backend cần NumPy, pandas, scikit-learn, XGBoost và joblib tương thích với môi
trường đã chạy thành công; không suy ra phiên bản huấn luyện từ metadata đóng gói.
