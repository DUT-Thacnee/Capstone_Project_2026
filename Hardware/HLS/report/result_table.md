**Bảng kết quả tóm tắt (dùng cho trực quan)**

Bảng dưới đây tóm tắt các chỉ số chính trích từ các report Vivado/HLS và benchmark summary. File CSV nguồn: [result_table.csv](result_table.csv)

| Cấu hình | DSPs | BRAM36 | LUTs | FFs | Total Power (W) | Dynamic Power (W) | Worst Slack (ns) | Latency Mean (ms) | Throughput (img/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 14x14 | 157 | 6 | 34,872 | 33,588 | 3.345 | 2.651 | 2.694 | n/a | n/a |
| 30x30 | 157 | 6 | 32,534 | 32,956 | 3.283 | 2.589 | 2.694 | 23.31 | 42.90 |
| nothing | 0 | 0 | 0 | 0 | 2.899 | 2.208 | n/a | n/a | n/a |

Ghi chú:
- Các giá trị tài nguyên (LUTs/FFs/DSP/BRAM) và công suất được lấy từ báo cáo Vivado trong `Hardware/HLS/report/*`.
- Latency/throughput là kết quả end‑to‑end tóm tắt trong `benchmark_summary/summary.json` (được ghi cho cấu hình FPGA tổng thể trong báo cáo hiện tại).

Muốn tôi vẽ biểu đồ PNG (bar charts) từ `result_table.csv` không? Tôi có thể tạo hình cho LUTs, Power và Latency/Throughput nhanh.
             