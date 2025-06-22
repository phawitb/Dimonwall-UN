import pandas as pd

# โหลดข้อมูล
df = pd.read_csv("Mockdata_Clean.csv")

# เพิ่มคอลัมน์ id ด้วยรูปแบบ U0001, U0002, ...
df.insert(0, "id", ["U" + str(i+1).zfill(4) for i in range(len(df))])

# บันทึกกลับลงไฟล์เดิม (หรือชื่อใหม่ถ้าต้องการสำรอง)
df.to_csv("Mockdata_Clean.csv", index=False, encoding='utf-8-sig')

print("✅ Field 'id' added successfully.")
