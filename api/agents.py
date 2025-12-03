from typing import Optional

import schema


# cv_file_path can be None if the user didn't upload a CV
def chat(req: schema.ChatRequest, cv_file_path: Optional[str]) -> str:
    pass
