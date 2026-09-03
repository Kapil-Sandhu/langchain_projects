from typing import TypedDict

# creating a type dictionary
# type dictionary is used to give you object feature a datatype so that,
# later others can understand what is the datatype of different object features
# but there is a main problem is that TypeDict is not a validator, 
# that means you can still you different datatype other then then which you define

class customer(TypedDict):
    name : str
    age : int
    number : int
    address : str
    

user_1: customer = {
    'name': "Kapil",
    'age' : 23,
    'number' : 1234567890,
    'address' : "Xyz 00"
}

print(user_1)
user_1['age'] = "25" 
# data type of age is changed but code will still run because not Typedic not use a validator

print(user_1)