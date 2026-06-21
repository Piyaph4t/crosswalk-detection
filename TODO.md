# TODO List - Crosswalk Detection Project

## 🧪 Testing & Validation (ลำดับความสำคัญสูง)

- [ ] รัน unit tests ทั้งหมดให้ผ่าน
  ```bash
  uv run pytest tests/ -v --cov=src/providers
  ```
- [ ] ทดสอบ config loading ว่าทำงานถูกต้อง
- [ ] ทดสอบ provider instantiation แต่ละประเภท

## 🔌 Hardware Integration Testing

- [ ] ทดสอบ USB webcam provider กับกล้องจริง
- [ ] ทดสอบ PiCamera provider บน Raspberry Pi (ถ้ามี CSI camera)
- [ ] ทดสอบ RTSP provider กับ CCTV stream จริง
- [ ] ทดสอบ reconnection logic (ถอดสาย USB/ปิด RTSP แล้วเปิดใหม่)

## 📊 Performance Benchmarking

- [ ] วัด FPS แต่ละ provider บน Raspberry Pi 4
- [ ] เปรียบเทียบว่าได้ตาม target 5-10 FPS หรือไม่
- [ ] วัด latency ของ RTSP stream
- [ ] วัด memory usage ขณะรัน inference

## 📝 Documentation

- [ ] เขียน deployment guide สำหรับ Raspberry Pi
- [ ] เขียนวิธี config แต่ละ provider ใน README
- [ ] เขียน troubleshooting guide (connection issues, performance issues)
- [ ] Document expected FPS สำหรับแต่ละ provider

## 🐛 Bug Fixes & Improvements

- [ ] แก้ Pyright warnings ที่เจอ (optional member access, import resolution)
- [ ] เพิ่ม file handler ให้ logging (เก็บ logs ไว้ดู)
- [ ] Implement video file provider ให้เสร็จ (สำหรับ testing)

## 📈 Future Enhancements

- [ ] เพิ่ม metrics collection (frame drops, reconnection count, avg FPS)
- [ ] เพิ่ม health check endpoint (ถ้าจะ deploy production)
- [ ] พิจารณา frame buffering สำหรับ RTSP (ถ้ามี jitter)
- [ ] เพิ่ม recording failed frames สำหรับ debugging

## 🔍 Code Review (รอ credit)

- [ ] Review provider architecture
- [ ] Review integration & tests
- [ ] Fix issues จาก review feedback

---

**ขั้นตอนแรกที่แนะนำ:** รัน `uv run pytest tests/ -v` เช็คว่า tests ผ่านหรือไม่
