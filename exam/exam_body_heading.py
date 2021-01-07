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

