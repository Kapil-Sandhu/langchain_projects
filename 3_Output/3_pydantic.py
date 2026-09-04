from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr, Field
from typing import Optional

load_dotenv()

# pytdantic is used most of the time when we work or an api because it validated the schema so that we dont face any
# error due to datatype of the schema

class Student(BaseModel):
    name : str = "Demo_Student" # this is used to set the default value for the object features
    age : int = None # both work the same with default value or use optional 
    roll_no : Optional[int] = None, # this is the way to set some feature as an optional value
    
    email : Optional[EmailStr] = None, # can not set a default email address
    
    cgpa : float = [ Field ( default=0.0, gt=0, lt=10, description="value of student cgpa") ]  # Field is used to set the condition on the input
    # description is similar to the Annotation in TypeDict
new_student = {
    # model_config = ConfigDict(validate_assignment=True, strict=True) for activation the object update validation
    'name' : 'Kapil',
    'age' : 23,
    #'age' : "24", # pydantic model automatically convert this into int because this can be easily converted to int, 
    # you can check the data type of student.age
    'roll_no' : 1,
    'email': "kapil@gmail.com",
    'cgpa' : 9.5
} 

demo_student = {}
student_0 = Student(**demo_student)
print(student_0)


student_1 = Student(**new_student)

print(student_1)
print( type(student_1.age))

#student_1.age = "25" # this should give an error because of pydantic schema validation
# but the validation only happends at the time of object creation by default not at the time of update 
# to force the pydantic model to perform validation at the time of update set 
#model_config = ConfigDict(validate_assignment=True, strict=True) int the class Student
#print(student_1) 
#print( type(student_1.age))

student_2 = Student(**{
    'name' : "Anshul"
})

print(student_2)
print(student_2.email)

# we can stock this object into dic or json formats also 
student_dict = dict(student_1)
print(student_1)
print(student_dict['cgpa']) # now i can fatch single value of object from dict

print(student_1.model_dump_json())