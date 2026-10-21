# Bank Customer Churn Prediction — Task 7

Ứng dụng Streamlit dự đoán khả năng khách hàng rời bỏ ngân hàng bằng model
XGBoost đã huấn luyện. Dashboard hỗ trợ nhập một khách hàng, dự đoán hàng loạt
từ CSV và giải thích dự đoán cá nhân bằng SHAP Waterfall theo đơn vị raw output.

Logic suy luận được dùng chung qua `inference.py`, có thể import vào Python hoặc
backend mà không khởi chạy Streamlit. Threshold phân loại cố định là **0.28**.

## 1. Chạy ứng dụng trên Windows

Môi trường đã được người dùng xác nhận hoạt động là Python **3.12.9**. Với
`.venv` đang có trong project, chạy từ PowerShell:

```powershell
Set-Location D:\Project
& .\.venv\Scripts\python.exe --version
& .\.venv\Scripts\python.exe -m streamlit run app.py
```

Mở địa chỉ Streamlit in trong terminal, thường là `http://localhost:8501`.
Dừng server bằng `Ctrl+C`. Các lệnh dùng trực tiếp interpreter của `.venv`,
không yêu cầu kích hoạt môi trường bằng `Activate.ps1`.

Nếu cài project trên máy mới và chưa có `.venv`, dùng Python 3.12 đã cài trên máy:

```powershell
python --version
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` ghim tám dependency trực tiếp cần cho Streamlit, inference
và unit test, theo metadata package thực tế trong `.venv/Lib/site-packages` của
project. `unittest` thuộc thư viện chuẩn Python nên không cần cài riêng. FastAPI
và Uvicorn chỉ cần cho adapter API tùy chọn, không nằm trong requirements này.

Đây chưa phải lockfile đầy đủ: dependency gián tiếp vẫn do pip giải quyết.
Chưa kiểm thử cài mới từ requirements trên mọi nền tảng. Phiên bản trong file
này là môi trường hiện tại; phiên bản trong metadata bundle là môi trường
đóng gói. Cả hai không phải bằng chứng về phiên bản huấn luyện.

Giữ `app.py`, `inference.py`, `best_xgb.joblib` và `feature_columns.joblib` cùng
thư mục để chạy dashboard. App hiện nạp trực tiếp hai artifact gốc; bundle
portable phục vụ đường tích hợp Python/backend.

## 2. Cấu trúc file

| File | Vai trò |
|---|---|
| `app.py` | Giao diện Streamlit, form, CSV, kết quả, SHAP Waterfall và bảng SHAP |
| `inference.py` | Validation, encoding, nạp artifact/bundle, prediction, mức rủi ro và threshold dùng chung |
| `best_xgb.joblib` | Model `XGBClassifier` đã huấn luyện; không được huấn luyện lại khi chạy app hoặc đóng gói |
| `feature_columns.joblib` | Danh sách 11 feature và thứ tự đầu vào model |
| `inference_bundle_portable.joblib` | Bundle v2 nhúng byte của hai artifact, schema, checksum, threshold và metadata |
| `package_inference.py` | Tạo bundle mới, mặc định portable; từ chối ghi đè file đã tồn tại |
| `requirements.txt` | Tám dependency trực tiếp được ghim theo phiên bản trong `.venv` |
| `.gitignore` | Loại trừ môi trường, cache, secrets, file tạm, CSV cục bộ và bundle v1 cũ |
| `.gitattributes` | Giữ LF cho Python và xử lý `.joblib` ở dạng binary |
| `tests/test_inference.py` | 13 unittest kiểm tra suy luận và đóng gói |
| `tests/fixtures/legacy_app_core.py` | Logic trước khi tách module, chỉ dùng làm chuẩn so sánh trong test |
| `INFERENCE.md` | Hướng dẫn chi tiết về module, bundle và backend |
| `.venv/` | Môi trường Python cục bộ; tạo lại khi chuyển máy thay vì sao chép nguyên môi trường |

`inference_bundle.joblib` v1 và `customers_test.csv` là file cục bộ bị ignore,
không thuộc bộ file bàn giao. Chỉ sử dụng các artifact hiện hành trong bảng trên.

## 3. Nhập dữ liệu và đọc kết quả

### Một khách hàng

Nhập thông tin trong form và bấm **Dự đoán nguy cơ rời bỏ**. Mỗi lần submit,
app mã hóa dữ liệu mới, gọi model và tính lại SHAP cho khách hàng đó. Chỉnh ô
nhập liệu chưa làm dự đoán tự cập nhật cho đến khi bấm nút.

| Trường đầu vào | Giá trị được chấp nhận |
|---|---|
| `CreditScore` | Số hữu hạn từ 300 đến 850 |
| `Geography` | Chính xác `France`, `Germany` hoặc `Spain` |
| `Gender` | Chính xác `Female` hoặc `Male` |
| `Age` | Số hữu hạn từ 18 đến 100 |
| `Tenure` | Số nguyên từ 0 đến 10 |
| `Balance` | Số hữu hạn, không âm |
| `NumOfProducts` | Số nguyên từ 1 đến 4 |
| `HasCrCard` | 0 hoặc 1 |
| `IsActiveMember` | 0 hoặc 1 |
| `EstimatedSalary` | Số hữu hạn, không âm |

Form dùng bước nhập số nguyên cho CreditScore và Age; validator dùng chung hiện
chỉ kiểm tra phạm vi và tính hữu hạn của hai trường này. Validator bắt buộc số
nguyên cho Tenure, NumOfProducts, HasCrCard và IsActiveMember. Không tự điền giá
trị mặc định cho trường bị thiếu trong CSV hoặc payload backend. Giá trị trống,
NaN, vô hạn và giá trị không hợp lệ bị từ chối.

Nhãn churn và mức rủi ro là hai kết quả có quy tắc riêng:

| Kết quả | Quy tắc |
|---|---|
| Nhãn churn | `1` nếu xác suất `>= 0.28`, ngược lại là `0` |
| Mức rủi ro Thấp | Xác suất `< 0.30` |
| Mức rủi ro Trung bình | Xác suất từ `0.30` đến dưới `0.60` |
| Mức rủi ro Cao | Xác suất `>= 0.60` |

Vì vậy, khoảng `[0.28, 0.30)` có cảnh báo churn nhưng mức rủi ro vẫn là Thấp.
Đây là hành vi hiện có của ứng dụng. Xác suất dự đoán không khẳng định chắc
chắn khách hàng sẽ rời bỏ.

### Upload CSV

Ví dụ nội dung `customers.csv`:

```csv
CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary
650,France,Female,40,5,75000,2,1,1,100000
700,Germany,Male,45,7,120000,1,1,0,85000
```

Tải file trong phần **Dự đoán hàng loạt từ CSV**. App đọc bằng `pd.read_csv()`,
kiểm tra toàn bộ dữ liệu, rồi dự đoán theo batch. CSV phải có đủ 10 cột theo tên;
thứ tự cột đầu vào có thể khác ví dụ. Cột bổ sung không đưa vào model nhưng được
giữ trong kết quả. Nếu có dòng không hợp lệ, toàn bộ batch bị từ chối; app chưa
có chế độ bỏ qua riêng các dòng lỗi.

Bảng và file tải xuống có thêm `ChurnProbability`, `PredictedChurn`, `RiskLevel`.
CSV xuất dùng UTF-8 có BOM (`utf-8-sig`). Luồng CSV hiện chưa có SHAP cho từng dòng.

### SHAP Waterfall

Biểu đồ nằm phía trên bảng 8 feature có độ lớn SHAP cao nhất. Waterfall hiển thị
11 feature, đi từ giá trị nền `E[f(X)]` đến đầu ra `f(x)` của khách hàng. Explainer
dùng `model_output="raw"`; các đóng góp SHAP là **raw output**, không phải phần
trăm hoặc điểm phần trăm xác suất. Giá trị dương đẩy raw output theo hướng churn;
giá trị âm đẩy theo hướng ở lại.

Explainer được cache nhưng SHAP được tính lại trên dữ liệu khách hàng mới sau
mỗi lần dự đoán. Lỗi vẽ biểu đồ được báo riêng và không làm mất kết quả dự đoán;
bảng SHAP vẫn được tạo nếu phần tính SHAP đã thành công.

## 4. Preprocessing và phạm vi hỗ trợ

`inference.py` kiểm tra dữ liệu, chuyển các trường số bằng `pd.to_numeric()` và
mã hóa Geography/Gender theo quy tắc:

| Giá trị | Feature tương ứng |
|---|---|
| France | `Geography_Germany=0`, `Geography_Spain=0` |
| Germany | `Geography_Germany=1`, `Geography_Spain=0` |
| Spain | `Geography_Germany=0`, `Geography_Spain=1` |
| Female | `Gender_Male=0` |
| Male | `Gender_Male=1` |

Đầu ra luôn có đúng thứ tự 11 feature:

```text
CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard,
IsActiveMember, EstimatedSalary, Geography_Germany, Geography_Spain, Gender_Male
```

Không thêm Scaler vào đường suy luận. Project chưa có notebook huấn luyện nên
chưa thể xác minh độc lập toàn bộ preprocessing lúc train hoặc phiên bản package
lúc huấn luyện. Kết quả test xác nhận logic hiện tại trước/sau khi tách module
giống nhau; không thay thế việc đối chiếu notebook.

Quốc gia ngoài France/Germany/Spain bị từ chối. Chỉ thêm quốc gia vào form hoặc
validation không giúp model học được quốc gia mới; muốn mở rộng cần dữ liệu và
đánh giá một phiên bản model phù hợp.

## 5. Dùng model từ Python hoặc backend/API

### Gọi trực tiếp từ Python

Đặt module trong đường dẫn import của backend. Ví dụ sau chạy từ thư mục project:

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
```

Kết quả cho bản ghi mặc định đã được ghi nhận trong lần kiểm thử runtime trước:

```json
{
  "churn_probability": 0.18973834812641144,
  "predicted_churn": 0,
  "risk_level": "Thấp",
  "threshold": 0.28
}
```

Đây là kết quả đã ghi nhận trong môi trường được kiểm thử, không phải giá trị
hardcode. Hàm trả về kiểu Python tương thích JSON. Gọi `predict_customer(customer)`
không truyền bundle sẽ dùng hai artifact cạnh `inference.py`; có thể chỉ định
`artifact_dir=...`. Không truyền đồng thời `artifact_dir` và `bundle_path`.

Backend nên dùng đường dẫn bundle tuyệt đối. Model được nạp lazily và cache theo
đường dẫn trong từng process; khởi động lại service khi triển khai release mới.

### Ví dụ adapter FastAPI

Project hiện **chưa có server API**. Nếu muốn tạo adapter, lưu ví dụ này thành
`api.py` cạnh module và bundle. Phần kiểm tra lúc startup giúp phát hiện sớm lỗi
artifact; các request dùng hàm suy luận chung:

```python
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from inference import load_bundle, predict_customer

BUNDLE = Path(__file__).resolve().parent / "inference_bundle_portable.joblib"

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_bundle(BUNDLE)
    yield

api = FastAPI(lifespan=lifespan)

@api.post("/predict")
def predict(payload: dict):
    try:
        return predict_customer(payload, bundle_path=BUNDLE)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
```

Cài và chạy adapter sau khi đã tạo `api.py`:

```powershell
& .\.venv\Scripts\python.exe -m pip install fastapi uvicorn
& .\.venv\Scripts\python.exe -m uvicorn api:api --host 127.0.0.1 --port 8000
```

Mở `http://127.0.0.1:8000/docs`, gửi `POST /predict` với JSON có 10 trường như
ví dụ Python. FastAPI/Uvicorn là dependency tùy chọn, không cần để chạy Streamlit
hoặc import `inference.py`. Adapter là ví dụ tích hợp, chưa được triển khai hay
kiểm thử HTTP trong project này.

## 6. Bundle portable và triển khai sang máy khác

Để sử dụng bundle portable v2, chép `inference.py`,
`inference_bundle_portable.joblib` và `requirements.txt` sang môi trường đích,
tạo môi trường Python rồi cài bằng `python -m pip install -r requirements.txt`.
Không cần chép riêng hai artifact gốc cho đường suy luận này. Bundle chứa nguyên
byte của artifact, không nạp rồi serialize lại đối tượng XGBoost, và không chứa
đường dẫn artifact trên máy nguồn.

Loader kiểm tra định dạng, schema, thứ tự feature, threshold, checksum module và
checksum của cả hai phần dữ liệu nhúng trước khi nạp model. Giữ nguyên byte của
`inference.py`, bao gồm kiểu xuống dòng, để khớp checksum. Chỉ nạp file joblib từ
nguồn đáng tin cậy; checksum không thay thế việc tin cậy nguồn file.

`.gitattributes` giữ LF cho file Python khi checkout, kể cả khi Git trên Windows
bật `core.autocrlf=true`, và đánh dấu `.joblib` là binary. Hãy giữ file cấu hình
này trong bộ file bàn giao; không tự đổi BOM hoặc kiểu xuống dòng của module.

Tạo bundle mới bằng tên chưa tồn tại:

```powershell
& .\.venv\Scripts\python.exe package_inference.py --mode portable --output inference_bundle_portable_release.joblib
```

Mặc định là portable. Luôn truyền `--output` với tên mới như ví dụ để tránh
trùng artifact. Nếu bỏ tùy chọn này, script chọn tên `inference_bundle.joblib`,
có thể trùng file v1 cục bộ; script từ chối ghi đè nếu file đích đã tồn tại.

Chế độ `--mode reference` chỉ lưu tham chiếu: cùng ổ thì đường dẫn tương đối,
khác ổ Windows thì đường dẫn tuyệt đối có đánh dấu. Tham chiếu tuyệt đối không
portable; chuyển máy cần tạo gói mới hoặc dùng portable. Cả hai chế độ đều giữ
checksum và từ chối ghi đè. Xem thêm [INFERENCE.md](INFERENCE.md).

## 7. Kiểm thử và kết quả đã xác nhận

Chạy toàn bộ unittest mà không khởi động Streamlit:

```powershell
Set-Location D:\Project
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Kết quả gần nhất **do người dùng chạy thực tế trên Windows**, được cung cấp
trong phiên làm việc:

```text
Ran 13 tests in 5.410s

OK
```

| Test | Kết quả |
|---|---|
| `test_all_valid_cases_match_original_encoding_and_probabilities` | PASS |
| `test_batch_preserves_index_extra_columns_and_predictions` | PASS |
| `test_bundle_rejects_changed_schema_threshold_and_checksum` | PASS |
| `test_bundle_round_trip_and_exclusive_creation` | PASS |
| `test_classification_threshold_and_original_risk_boundaries` | PASS |
| `test_default_feature_names_order_and_values` | PASS |
| `test_import_has_no_streamlit_or_artifact_loading` | PASS |
| `test_invalid_cases_preserve_error_messages` | PASS |
| `test_portable_bundle_relocates_without_reading_original_artifacts` | PASS |
| `test_predict_customer_matches_original_and_is_json_serializable` | PASS |
| `test_reference_bundle_cross_drive_fallback_loads_and_checks_hash` | PASS |
| `test_reference_bundle_same_drive_keeps_relative_paths` | PASS |
| `test_windows_different_drive_reference_path` | PASS |

Các test đối chiếu encoding, xác suất và thông báo validation với fixture logic
cũ; dùng model thật cho so sánh dự đoán. Test biên phân loại dùng thêm model giả
trả xác suất cố định để kiểm tra chính xác các mốc. Test khác ổ có cả ngữ nghĩa
đường dẫn Windows (`ntpath`) và mô phỏng lỗi `relpath()` để kiểm tra fallback.

Trước khi tách module, người dùng cũng đã chạy bộ 34 ca encoder/model và nhận
34 PASS. Hai bộ có nội dung chồng lặp nên không cộng thành số test độc lập.
Người dùng đã xác nhận Streamlit và SHAP Waterfall hiển thị thành công; 13 unittest
không phải bộ kiểm thử giao diện. Chưa có kết quả kiểm thử HTTP hoặc triển khai
cloud. Các test hồi quy này cũng không đo chất lượng dự đoán như AUC hay recall.

## 8. Xử lý một số lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `No Python at ...` | `.venv` trỏ tới interpreter không khả dụng; kiểm tra Python trên máy và tạo môi trường mới phù hợp |
| Thiếu module, đặc biệt `matplotlib` | Cài dependency bằng chính `.venv\Scripts\python.exe -m pip`, rồi chạy lại app |
| Cảnh báo XGBoost về serialized model phiên bản cũ | Cảnh báo đã xuất hiện trong lần chạy 13/13 PASS; không bị coi là lỗi test, nhưng không bảo đảm tương thích với mọi phiên bản khác |
| Checksum `inference.py` không khớp | Dùng đúng module đi kèm bundle hoặc tạo bundle tên mới từ module hiện tại; không bỏ kiểm tra checksum |
| `Refusing to overwrite` | Chọn tên output mới khi đóng gói |
| CSV bị từ chối | Kiểm tra đủ tên cột, category đúng chữ hoa/thường, giá trị thiếu và phạm vi của tất cả dòng |

Project có `requirements.txt` ghim dependency trực tiếp, nhưng chưa có notebook
huấn luyện, lockfile đầy đủ cho dependency gián tiếp hoặc cấu hình triển khai
cloud. Model, schema và threshold hiện tại được giữ nguyên;
không có bước huấn luyện lại khi chạy app, gọi API mẫu hoặc tạo bundle.
