# Bomber_hcmus_2026
1. Cần cải thiện trường hợp khi agent rơi vào trạng thái gặp danger_soon:
   - Không trả về danger_soon nữa mà trả về danger_time (còn bao nhiêu lượt tìm bom nổ) và danger_steps (bom cách vị trí hiện tại của agent bao nhiêu bước)
     
2. Cần tạo thêm hàm để agent xử lý trường hợp gặp chain reaction (bom chồng bom)
  
3. Cần cải thiện hàm move_to_targets vì hàm đang được code để tiếp cận trực tiếp địch -> dễ bị dính bẫy dồn tường
   - check hoặc dự đoán trước địch có cùng hàng hay cột không chỉ tấn công địch từ xa, hạn chế tiếp cận khi không cần thiết

4. Cần cải thiện việc đặt bom, khi nào cần đặt và khi nào không để tránh việc lãng phí bom cho các trường hợp không cần thiết

5. Cần lưu lịch sử 4 bước gần nhất của agent để tránh lặp lại

6. tranh items

Note: Hiện tại các team khi tới step nhất định bị lặp lại bước -> tối ưu của team không cho lặp lại + xem status địch để đi càng xa địch càng tốt + lụm items và phá thùng để được + điểm phần khác
