- Tìm hiểu về thuật toán MVS (Multi-View Stereo) để tạo ra đám mây điểm dày đặc từ đám mây điểm thưa

- Tạo lưới bề mặt (mesh) từ đám mây điểm:
    - Sử dụng thuật toán như Poisson Surface Reconstruction để tạo mesh từ đám mây điểm.
    - Sử dụng MeshLab hoặc một thư viện như PCL (Point Cloud Library) để thực hiện bước này.

- Tạo kết cấu (texturing) cho mesh:
    - Ánh xạ màu sắc từ hình ảnh gốc lên mesh 3D để có một mô hình có kết cấu thực tế.
    - Thư viện như MVS-Texturing

- Tinh chỉnh mô hình:
    - Loại bỏ nhiễu, lấp đầy lỗ hổng, và làm mịn bề mặt mô hình.