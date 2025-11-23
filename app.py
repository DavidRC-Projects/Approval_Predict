from app_pages.multipage import MultiPage

from app_pages.page_summary import page_summary_body
from app_pages.page_feature_impact import page_feature_impact_body
from app_pages.page_approval_predictor import page_approval_predictor_body
from app_pages.page_project_findings import page_project_findings_body
from app_pages.page_final_model import page_final_model_body

app = MultiPage(app_name="Approval Predictor Dashboard")

app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Feature Impact Study", page_feature_impact_body)
app.add_page("Approval Predictor", page_approval_predictor_body)
app.add_page("ML Model Performance", page_final_model_body)
app.add_page("Project Hypotheses and Conclusions", page_project_findings_body)

app.run()
