# hack to hook modules in subfolder
import os
import sys
app_root = os.path.dirname(os.path.dirname(__file__))
app_root_source = app_root + "/source"
app_root_exam = app_root + "/exam"
if app_root not in sys.path:
    sys.path.append(app_root)
if app_root_source not in sys.path:
    sys.path.append(app_root_source)
if app_root_exam not in sys.path:
    sys.path.append(app_root_exam)


from exam.exam_data import read_rdii_csv

pd_acce = read_rdii_csv("dan_body1/acce.txt")
pd_gyro = read_rdii_csv("dan_body1/gyro.txt")
pd_orit = read_rdii_csv("dan_body1/orientation.txt")

print(pd_acce)
print(pd_gyro)
print(pd_orit)
