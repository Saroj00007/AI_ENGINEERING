from pydantic import BaseModel , Field


class ReposageExplanation(BaseModel):
    summary : str = Field(description="Summary of the code") 
    
    releent_file : list[str] = Field(description="List of relevant file names")
    
    uncertanities : list[str] = Field(description="List of uncertanities")



