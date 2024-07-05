from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.sql import SparkSession
import streamlit as st
import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = SparkSession.builder.appName("dropout_scoring_analytics").getOrCreate()

model = RandomForestClassificationModel.load("model")

# Streamlit app title
st.title("Student Dropout Prediction")

# Define lists for input features
marital_status_list = [
    "single",
    "married",
    "widower",
    "divorced",
    "facto union",
    "legally separated",
]

application_mode_list = [
    "1st phase - general contingent",
    "Ordinance No. 612/93",
    "1st phase - special contingent (Azores Island)",
    "Holders of other higher courses",
    "Ordinance No. 854-B/99",
    "International student (bachelor)",
    "1st phase - special contingent (Madeira Island)",
    "2nd phase - general contingent",
    "3rd phase - general contingent",
    "Ordinance No. 533-A/99, item b2 (Different Plan)",
    "Ordinance No. 533-A/99, item b3 (Other Institution)",
    "Over 23 years old",
    "Transfer",
    "Change of course",
    "Technological specialization diploma holders",
    "Change of institution/course",
    "Short cycle diploma holders",
    "Change of institution/course (International)",
]

application_order_list = [
    "first choice",
    "second choice",
    "third choice",
    "fourth choice",
    "fifth choice",
    "sixth choice",
    "seventh choice",
    "eighth choice",
    "ninth choice",
    "last choice",
]

course_list = [
    "Biofuel Production Technologies",
    "Animation and Multimedia Design",
    "Social Service (evening attendance)",
    "Agronomy",
    "Communication Design",
    "Veterinary Nursing",
    "Informatics Engineering",
    "Equinculture",
    "Management",
    "Social Service",
    "Tourism",
    "Nursing",
    "Oral Hygiene",
    "Advertising and Marketing Management",
    "Journalism and Communication",
    "Basic Education",
    "Management (evening attendance)",
]

daytime_evening_attendance_list = ["daytime", "evening"]

previous_qualification_list = [
    "Secondary education",
    "Higher education - bachelor's degree",
    "Higher education - degree",
    "Higher education - master's",
    "Higher education - doctorate",
    "Frequency of higher education",
    "12th year of schooling - not completed",
    "11th year of schooling - not completed",
    "Other - 11th year of schooling",
    "10th year of schooling",
    "10th year of schooling - not completed",
    "Basic education 3rd cycle (9th/10th/11th year) or equiv.",
    "Basic education 2nd cycle (6th/7th/8th year) or equiv.",
    "Technological specialization course",
    "Higher education - degree (1st cycle)",
    "Professional higher technical course",
    "Higher education - master (2nd cycle)",
]

nacionality_list = [
    "Portuguese",
    "German",
    "Spanish",
    "Italian",
    "Dutch",
    "English",
    "Lithuanian",
    "Angolan",
    "Cape Verdean",
    "Guinean",
    "Mozambican",
    "Santomean",
    "Turkish",
    "Brazilian",
    "Romanian",
    "Moldova (Republic of)",
    "Mexican",
    "Ukrainian",
    "Russian",
    "Cuban",
    "Colombian",
]
mother_qualification_list = [
    "Secondary Education - 12th Year of Schooling or Eq.",
    "Higher Education - Bachelor's Degree",
    "Higher Education - Degree",
    "Higher Education - Master's",
    "Higher Education - Doctorate",
    "Frequency of Higher Education",
    "12th Year of Schooling - Not Completed",
    "11th Year of Schooling - Not Completed",
    "7th Year (Old)",
    "Other - 11th Year of Schooling",
    "10th Year of Schooling",
    "General commerce course",
    "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
    "Technical-professional course",
    "7th year of schooling",
    "2nd cycle of the general high school course",
    "9th Year of Schooling - Not Completed",
    "8th year of schooling",
    "Unknown",
    "Can't read or write",
    "Can read without having a 4th year of schooling",
    "Basic education 1st cycle (4th/5th year) or equiv.",
    "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
    "Technological specialization course",
    "Higher education - degree (1st cycle)",
    "Specialized higher studies course",
    "Professional higher technical course",
    "Higher Education - Master (2nd cycle)",
    "Higher Education - Doctorate (3rd cycle)",
]

father_qualification_list = [
    "Secondary Education - 12th Year of Schooling or Eq.",
    "Higher Education - Bachelor's Degree",
    "Higher Education - Degree",
    "Higher Education - Master's",
    "Higher Education - Doctorate",
    "Frequency of Higher Education",
    "12th Year of Schooling - Not Completed",
    "11th Year of Schooling - Not Completed",
    "7th Year (Old)",
    "Other - 11th Year of Schooling",
    "10th Year of Schooling",
    "General commerce course",
    "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
    "Technical-professional course",
    "7th year of schooling",
    "2nd cycle of the general high school course",
    "9th Year of Schooling - Not Completed",
    "8th year of schooling",
    "Unknown",
    "Can't read or write",
    "Can read without having a 4th year of schooling",
    "Basic education 1st cycle (4th/5th year) or equiv.",
    "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
    "Technological specialization course",
    "Higher education - degree (1st cycle)",
    "Specialized higher studies course",
    "Professional higher technical course",
    "Higher Education - Master (2nd cycle)",
    "Higher Education - Doctorate (3rd cycle)",
]

mother_occupation_list = [
    "Student",
    "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers",
    "Specialists in Intellectual and Scientific Activities",
    "Intermediate Level Technicians and Professions",
    "Administrative staff",
    "Personal Services, Security and Safety Workers and Sellers",
    "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry",
    "Skilled Workers in Industry, Construction and Craftsmen",
    "Installation and Machine Operators and Assembly Workers",
    "Unskilled Workers",
    "Armed Forces Professions",
    "Other Situation",
    "(blank)",
    "Health professionals",
    "teachers",
    "Specialists in information and communication technologies (ICT)",
    "Intermediate level science and engineering technicians and professions",
    "Technicians and professionals, of intermediate level of health",
    "Intermediate level technicians from legal, social, sports, cultural and similar services",
    "Office workers, secretaries in general and data processing operators",
    "Data, accounting, statistical, financial services and registry-related operators",
    "Other administrative support staff",
    "personal service workers",
    "sellers",
    "Personal care workers and the like",
    "Skilled construction workers and the like, except electricians",
    "Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like",
    "Workers in food processing, woodworking, clothing and other industries and crafts",
    "cleaning workers",
    "Unskilled workers in agriculture, animal production, fisheries and forestry",
    "Unskilled workers in extractive industry, construction, manufacturing and transport",
    "Meal preparation assistants",
]

father_occupation_list = [
    "Student",
    "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers",
    "Specialists in Intellectual and Scientific Activities",
    "Intermediate Level Technicians and Professions",
    "Administrative staff",
    "Personal Services, Security and Safety Workers and Sellers",
    "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry",
    "Skilled Workers in Industry, Construction and Craftsmen",
    "Installation and Machine Operators and Assembly Workers",
    "Unskilled Workers",
    "Armed Forces Professions",
    "Other Situation",
    "(blank)",
    "Armed Forces Officers",
    "Armed Forces Sergeants",
    "Other Armed Forces personnel",
    "Directors of administrative and commercial services",
    "Hotel, catering, trade and other services directors",
    "Specialists in the physical sciences, mathematics, engineering and related techniques",
    "Health professionals",
    "Teachers",
    "Specialists in finance, accounting, administrative organization, public and commercial relations",
    "Intermediate level science and engineering technicians and professions",
    "Technicians and professionals, of intermediate level of health",
    "Intermediate level technicians from legal, social, sports, cultural and similar services",
    "Information and communication technology technicians",
    "Office workers, secretaries in general and data processing operators",
    "Data, accounting, statistical, financial services and registry-related operators",
    "Other administrative support staff",
    "Personal service workers",
    "Sellers",
    "Personal care workers and the like",
    "Protection and security services personnel",
    "Market-oriented farmers and skilled agricultural and animal production workers",
    "Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence",
    "Skilled construction workers and the like, except electricians",
    "Skilled workers in metallurgy, metalworking and similar",
    "Skilled workers in electricity and electronics",
    "Workers in food processing, woodworking, clothing and other industries and crafts",
    "Fixed plant and machine operators",
    "Assembly workers",
    "Vehicle drivers and mobile equipment operators",
    "Unskilled workers in agriculture, animal production, fisheries and forestry",
    "Unskilled workers in extractive industry, construction, manufacturing and transport",
    "Meal preparation assistants",
    "Street vendors (except food) and street service providers",
]

displaced_list = ["yes", "no"]
education_special_needs_list = ["yes", "no"]
debtor_list = ["yes", "no"]
tuition_fees_list = ["yes", "no"]
gender_list = ["male", "female"]
scholarship_holder_list = ["yes", "no"]
international_list = ["yes", "no"]

# Mapping for string features
marital_status_map = {
    "single": 1,
    "married": 2,
    "widower": 3,
    "divorced": 4,
    "facto union": 5,
    "legally separated": 6,
}

application_mode_map = {
    "1st phase - general contingent": 1,
    "Ordinance No. 612/93": 2,
    "1st phase - special contingent (Azores Island)": 5,
    "Holders of other higher courses": 7,
    "Ordinance No. 854-B/99": 10,
    "International student (bachelor)": 15,
    "1st phase - special contingent (Madeira Island)": 16,
    "2nd phase - general contingent": 17,
    "3rd phase - general contingent": 18,
    "Ordinance No. 533-A/99, item b2 (Different Plan)": 26,
    "Ordinance No. 533-A/99, item b3 (Other Institution)": 27,
    "Over 23 years old": 39,
    "Transfer": 42,
    "Change of course": 43,
    "Technological specialization diploma holders": 44,
    "Change of institution/course": 51,
    "Short cycle diploma holders": 53,
    "Change of institution/course (International)": 57,
}

application_order_map = {
    "first choice": 0,
    "second choice": 1,
    "third choice": 2,
    "fourth choice": 3,
    "fifth choice": 4,
    "sixth choice": 5,
    "seventh choice": 6,
    "eighth choice": 7,
    "ninth choice": 8,
    "last choice": 9,
}

course_map = {
    "Biofuel Production Technologies": 33,
    "Animation and Multimedia Design": 171,
    "Social Service (evening attendance)": 8014,
    "Agronomy": 9003,
    "Communication Design": 9070,
    "Veterinary Nursing": 9085,
    "Informatics Engineering": 9119,
    "Equinculture": 9130,
    "Management": 9147,
    "Social Service": 9238,
    "Tourism": 9254,
    "Nursing": 9500,
    "Oral Hygiene": 9556,
    "Advertising and Marketing Management": 9670,
    "Journalism and Communication": 9773,
    "Basic Education": 9853,
    "Management (evening attendance)": 9991,
}

daytime_evening_attendance_map = {"daytime": 1, "evening": 0}

previous_qualification_map = {
    "Secondary education": 1,
    "Higher education - bachelor's degree": 2,
    "Higher education - degree": 3,
    "Higher education - master's": 4,
    "Higher education - doctorate": 5,
    "Frequency of higher education": 6,
    "12th year of schooling - not completed": 9,
    "11th year of schooling - not completed": 10,
    "Other - 11th year of schooling": 12,
    "10th year of schooling": 14,
    "10th year of schooling - not completed": 15,
    "Basic education 3rd cycle (9th/10th/11th year) or equiv.": 19,
    "Basic education 2nd cycle (6th/7th/8th year) or equiv.": 38,
    "Technological specialization course": 39,
    "Higher education - degree (1st cycle)": 40,
    "Professional higher technical course": 42,
    "Higher education - master (2nd cycle)": 43,
}

nationality_map = {
    "Portuguese": 1,
    "German": 2,
    "Spanish": 6,
    "Italian": 11,
    "Dutch": 13,
    "English": 14,
    "Lithuanian": 17,
    "Angolan": 21,
    "Cape Verdean": 22,
    "Guinean": 24,
    "Mozambican": 25,
    "Santomean": 26,
    "Turkish": 32,
    "Brazilian": 41,
    "Romanian": 62,
    "Moldova (Republic of)": 100,
    "Mexican": 101,
    "Ukrainian": 103,
    "Russian": 105,
    "Cuban": 108,
    "Colombian": 109,
}

mother_qualification_map = {
    "Secondary Education - 12th Year of Schooling or Eq.": 1,
    "Higher Education - Bachelor's Degree": 2,
    "Higher Education - Degree": 3,
    "Higher Education - Master's": 4,
    "Higher Education - Doctorate": 5,
    "Frequency of Higher Education": 6,
    "12th Year of Schooling - Not Completed": 9,
    "11th Year of Schooling - Not Completed": 10,
    "7th Year (Old)": 11,
    "Other - 11th Year of Schooling": 12,
    "10th Year of Schooling": 14,
    "General commerce course": 18,
    "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.": 19,
    "Technical-professional course": 22,
    "7th year of schooling": 26,
    "2nd cycle of the general high school course": 27,
    "9th Year of Schooling - Not Completed": 29,
    "8th year of schooling": 30,
    "Unknown": 34,
    "Can't read or write": 35,
    "Can read without having a 4th year of schooling": 36,
    "Basic education 1st cycle (4th/5th year) or equiv.": 37,
    "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.": 38,
    "Technological specialization course": 39,
    "Higher education - degree (1st cycle)": 40,
    "Specialized higher studies course": 41,
    "Professional higher technical course": 42,
    "Higher Education - Master (2nd cycle)": 43,
    "Higher Education - Doctorate (3rd cycle)": 44,
}

father_qualification_map = {
    "Secondary Education - 12th Year of Schooling or Eq.": 1,
    "Higher Education - Bachelor's Degree": 2,
    "Higher Education - Degree": 3,
    "Higher Education - Master's": 4,
    "Higher Education - Doctorate": 5,
    "Frequency of Higher Education": 6,
    "12th Year of Schooling - Not Completed": 9,
    "11th Year of Schooling - Not Completed": 10,
    "7th Year (Old)": 11,
    "Other - 11th Year of Schooling": 12,
    "10th Year of Schooling": 14,
    "General commerce course": 18,
    "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.": 19,
    "Technical-professional course": 22,
    "7th year of schooling": 26,
    "2nd cycle of the general high school course": 27,
    "9th Year of Schooling - Not Completed": 29,
    "8th year of schooling": 30,
    "Unknown": 34,
    "Can't read or write": 35,
    "Can read without having a 4th year of schooling": 36,
    "Basic education 1st cycle (4th/5th year) or equiv.": 37,
    "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.": 38,
    "Technological specialization course": 39,
    "Higher education - degree (1st cycle)": 40,
    "Specialized higher studies course": 41,
    "Professional higher technical course": 42,
    "Higher Education - Master (2nd cycle)": 43,
    "Higher Education - Doctorate (3rd cycle)": 44,
}

mother_occupation_map = {
    "Student": 0,
    "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers": 1,
    "Specialists in Intellectual and Scientific Activities": 2,
    "Intermediate Level Technicians and Professions": 3,
    "Administrative staff": 4,
    "Personal Services, Security and Safety Workers and Sellers": 5,
    "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry": 6,
    "Skilled Workers in Industry, Construction and Craftsmen": 7,
    "Installation and Machine Operators and Assembly Workers": 8,
    "Unskilled Workers": 9,
    "Armed Forces Professions": 10,
    "Other Situation": 90,
    "(blank)": 99,
    "Health professionals": 122,
    "teachers": 123,
    "Specialists in information and communication technologies (ICT)": 125,
    "Intermediate level science and engineering technicians and professions": 131,
    "Technicians and professionals, of intermediate level of health": 132,
    "Intermediate level technicians from legal, social, sports, cultural and similar services": 134,
    "Office workers, secretaries in general and data processing operators": 141,
    "Data, accounting, statistical, financial services and registry-related operators": 143,
    "Other administrative support staff": 144,
    "personal service workers": 151,
    "sellers": 152,
    "Personal care workers and the like": 153,
    "Skilled construction workers and the like, except electricians": 171,
    "Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like": 173,
    "Workers in food processing, woodworking, clothing and other industries and crafts": 175,
    "cleaning workers": 191,
    "Unskilled workers in agriculture, animal production, fisheries and forestry": 192,
    "Unskilled workers in extractive industry, construction, manufacturing and transport": 193,
    "Meal preparation assistants": 194,
}

father_occupation_map = {
    "Student": 0,
    "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers": 1,
    "Specialists in Intellectual and Scientific Activities": 2,
    "Intermediate Level Technicians and Professions": 3,
    "Administrative staff": 4,
    "Personal Services, Security and Safety Workers and Sellers": 5,
    "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry": 6,
    "Skilled Workers in Industry, Construction and Craftsmen": 7,
    "Installation and Machine Operators and Assembly Workers": 8,
    "Unskilled Workers": 9,
    "Armed Forces Professions": 10,
    "Other Situation": 90,
    "(blank)": 99,
    "Armed Forces Officers": 101,
    "Armed Forces Sergeants": 102,
    "Other Armed Forces personnel": 103,
    "Directors of administrative and commercial services": 112,
    "Hotel, catering, trade and other services directors": 114,
    "Specialists in the physical sciences, mathematics, engineering and related techniques": 121,
    "Health professionals": 122,
    "teachers": 123,
    "Specialists in finance, accounting, administrative organization, public and commercial relations": 124,
    "Intermediate level science and engineering technicians and professions": 131,
    "Technicians and professionals, of intermediate level of health": 132,
    "Intermediate level technicians from legal, social, sports, cultural and similar services": 134,
    "Information and communication technology technicians": 135,
    "Office workers, secretaries in general and data processing operators": 141,
    "Data, accounting, statistical, financial services and registry-related operators": 143,
    "Other administrative support staff": 144,
    "personal service workers": 151,
    "sellers": 152,
    "Personal care workers and the like": 153,
    "Protection and security services personnel": 154,
    "Market-oriented farmers and skilled agricultural and animal production workers": 161,
    "Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence": 163,
    "Skilled construction workers and the like, except electricians": 171,
    "Skilled workers in metallurgy, metalworking and similar": 172,
    "Skilled workers in electricity and electronics": 174,
    "Workers in food processing, woodworking, clothing and other industries and crafts": 175,
    "Fixed plant and machine operators": 181,
    "assembly workers": 182,
    "Vehicle drivers and mobile equipment operators": 183,
    "Unskilled workers in agriculture, animal production, fisheries and forestry": 192,
    "Unskilled workers in extractive industry, construction, manufacturing and transport": 193,
    "Meal preparation assistants": 194,
    "Street vendors (except food) and street service providers": 195,
}

displaced_map = {"yes": 1, "no": 0}

education_special_needs_map = {"yes": 1, "no": 0}

debtor_map = {"yes": 1, "no": 0}

tuition_fees_map = {"yes": 1, "no": 0}

gender_map = {"male": 1, "female": 0}

scholarship_holder_map = {"yes": 1, "no": 0}

international_map = {"yes": 1, "no": 0}

# Input fields for features using selectbox
marital_status = st.selectbox("Marital status", marital_status_list)
application_mode = st.selectbox("Application mode", application_mode_list)
application_order = st.selectbox("Application order", application_order_list)
course = st.selectbox("Course", course_list)
daytime_evening_attendance = st.selectbox(
    "Daytime/evening attendance", daytime_evening_attendance_list
)
previous_qualification = st.selectbox(
    "Previous qualification", previous_qualification_list
)
nationality = st.selectbox("Nationality", nacionality_list)
mother_qualification = st.selectbox("Mother's qualification", mother_qualification_list)
father_qualification = st.selectbox("Father's qualification", father_qualification_list)
mother_occupation = st.selectbox("Mother's occupation", mother_occupation_list)
father_occupation = st.selectbox("Father's occupation", father_occupation_list)
displaced = st.selectbox("Displaced", displaced_list)
education_special_needs = st.selectbox(
    "Education special needs", education_special_needs_list
)
debtor = st.selectbox("Debtor", debtor_list)
tuition_fees = st.selectbox("Tuition fees", tuition_fees_list)
gender = st.selectbox("Gender", gender_list)
scholarship_holder = st.selectbox("Scholarship holder", scholarship_holder_list)
international = st.selectbox("International", international_list)

previous_qualification_grade = st.number_input(
    "Previous qualification grade", min_value=0, max_value=200, value=0, step=1
)
admission_grade = st.number_input("Admission grade", min_value=0, value=0)
age_enrollment = st.number_input("Age at enrollment", min_value=0, value=0, step=1)
unemployment_rate = st.number_input("Unemployment rate", min_value=0.0, value=0.0)
inflation_rate = st.number_input("Inflation rate", min_value=0.0, value=0.0)
gdp = st.number_input("GDP", min_value=0.0, value=0.0)
curr_1st_credited = st.number_input(
    "Curricular units in 1st semester that are credited", min_value=0, value=0
)
curr_1st_enrolled = st.number_input(
    "Curricular units in 1st semester that are enrolled", min_value=0, value=0
)
curr_1st_eval = st.number_input(
    "Curricular units in 1st semester that are eval", min_value=0, value=0
)
curr_1st_approved = st.number_input(
    "Curricular units in 1st semester that are approved", min_value=0, value=0
)
curr_1st_grade = st.number_input(
    "Curricular units in 1st semester that are graded", min_value=0, value=0
)
curr_1st_without_eval = st.number_input(
    "Curricular units in 1st semester that are without eval", min_value=0, value=0
)
curr_2nd_credited = st.number_input(
    "Curricular units in 2nd semester that are credited", min_value=0, value=0
)
curr_2nd_enrolled = st.number_input(
    "Curricular units in 2nd semester that are enrolled", min_value=0, value=0
)
curr_2nd_eval = st.number_input(
    "Curricular units in 2nd semester that are eval", min_value=0, value=0
)
curr_2nd_approved = st.number_input(
    "Curricular units in 2nd semester that are approved", min_value=0, value=0
)
curr_2nd_grade = st.number_input(
    "Curricular units in 2nd semester that are graded", min_value=0, value=0
)
curr_2nd_without_eval = st.number_input(
    "Curricular units in 2nd semester that are without eval", min_value=0, value=0
)

if st.button("Predict"):
    data = [
        [
            debtor_map[debtor],
            gender_map[gender],
            scholarship_holder_map[scholarship_holder],
            international_map[international],
            tuition_fees_map[tuition_fees],
            education_special_needs_map[education_special_needs],
            displaced_map[displaced],
            father_occupation_map[father_occupation],
            mother_occupation_map[mother_occupation],
            father_qualification_map[father_qualification],
            mother_qualification_map[mother_qualification],
            nationality_map[nationality],
            previous_qualification_map[previous_qualification],
            daytime_evening_attendance_map[daytime_evening_attendance],
            course_map[course],
            application_order_map[application_order],
            application_mode_map[application_mode],
            marital_status_map[marital_status],
            previous_qualification_grade,
            admission_grade,
            inflation_rate,
            gdp,
            unemployment_rate,
            age_enrollment,
            curr_1st_credited,
            curr_1st_enrolled,
            curr_1st_eval,
            curr_1st_approved,
            curr_1st_grade,
            curr_1st_without_eval,
            curr_2nd_credited,
            curr_2nd_enrolled,
            curr_2nd_eval,
            curr_2nd_approved,
            curr_2nd_grade,
            curr_2nd_without_eval,
        ]
    ]

    columns = [
        "Debtor",
        "Gender",
        "Scholarship_holder",
        "International",
        "Tuition_fees_up_to_date",
        "Educational_special_needs",
        "Displaced",
        "Fathers_occupation",
        "Mothers_occupation",
        "Fathers_qualification",
        "Mothers_qualification",
        "Nacionality",
        "Previous_qualification",
        "Daytime_evening_attendance",
        "Course",
        "Application_order",
        "Application_mode",
        "Marital_status",
        "Previous_qualification_grade",
        "Admission_grade",
        "Inflation_rate",
        "GDP",
        "Unemployment_rate",
        "Age_at_enrollment",
        "Curricular_units_1st_sem_credited",
        "Curricular_units_1st_sem_enrolled",
        "Curricular_units_1st_sem_evaluations",
        "Curricular_units_1st_sem_approved",
        "Curricular_units_1st_sem_grade",
        "Curricular_units_1st_sem_without_evaluations",
        "Curricular_units_2nd_sem_credited",
        "Curricular_units_2nd_sem_enrolled",
        "Curricular_units_2nd_sem_evaluations",
        "Curricular_units_2nd_sem_approved",
        "Curricular_units_2nd_sem_grade",
        "Curricular_units_2nd_sem_without_evaluations",
    ]

    inputCols = [
        "Debtor",
        "Gender",
        "Scholarship_holder",
        "International",
        "Tuition_fees_up_to_date",
        "Educational_special_needs",
        "Displaced",
        "Fathers_occupation",
        "Mothers_occupation",
        "Fathers_qualification",
        "Mothers_qualification",
        "Nacionality",
        "Previous_qualification",
        "Daytime_evening_attendance",
        "Course",
        "Application_order",
        "Application_mode",
        "Marital_status",
        "scaled_Previous_qualification_grade",
        "scaled_Admission_grade",
        "scaled_Inflation_rate",
        "scaled_GDP",
        "scaled_Unemployment_rate",
        "scaled_Age_at_enrollment",
        "scaled_Curricular_units_1st_sem_credited",
        "scaled_Curricular_units_1st_sem_enrolled",
        "scaled_Curricular_units_1st_sem_evaluations",
        "scaled_Curricular_units_1st_sem_approved",
        "scaled_Curricular_units_1st_sem_grade",
        "scaled_Curricular_units_1st_sem_without_evaluations",
        "scaled_Curricular_units_2nd_sem_credited",
        "scaled_Curricular_units_2nd_sem_enrolled",
        "scaled_Curricular_units_2nd_sem_evaluations",
        "scaled_Curricular_units_2nd_sem_approved",
        "scaled_Curricular_units_2nd_sem_grade",
        "scaled_Curricular_units_2nd_sem_without_evaluations",
    ]

    df = spark.createDataFrame(data, columns)

    assembler1 = (
        VectorAssembler()
        .setInputCols(["Previous_qualification_grade"])
        .setOutputCol("vec_Previous_qualification_grade")
    )
    assembler2 = (
        VectorAssembler()
        .setInputCols(["Admission_grade"])
        .setOutputCol("vec_Admission_grade")
    )
    assembler3 = (
        VectorAssembler()
        .setInputCols(["Unemployment_rate"])
        .setOutputCol("vec_Unemployment_rate")
    )
    assembler4 = (
        VectorAssembler()
        .setInputCols(["Inflation_rate"])
        .setOutputCol("vec_Inflation_rate")
    )
    assembler5 = VectorAssembler().setInputCols(["GDP"]).setOutputCol("vec_GDP")
    assembler6 = (
        VectorAssembler()
        .setInputCols(["Age_at_enrollment"])
        .setOutputCol("vec_Age_at_enrollment")
    )
    assembler7 = (
        VectorAssembler()
        .setInputCols(["Curricular_units_1st_sem_credited"])
        .setOutputCol("vec_Curricular_units_1st_sem_credited")
    )
    assembler8 = (
        VectorAssembler()
        .setInputCols(["Curricular_units_1st_sem_enrolled"])
        .setOutputCol("vec_Curricular_units_1st_sem_enrolled")
    )
    assembler9 = (
        VectorAssembler()
        .setInputCols(["Curricular_units_1st_sem_evaluations"])
        .setOutputCol("vec_Curricular_units_1st_sem_evaluations")
    )
    assembler10 = (
        VectorAssembler()
        .setInputCols(["Curricular_units_1st_sem_approved"])
        .setOutputCol("vec_Curricular_units_1st_sem_approved")
    )
    assembler11 = (
        VectorAssembler()
        .setInputCols(["Curricular_units_1st_sem_grade"])
        .setOutputCol("vec_Curricular_units_1st_sem_grade")
    )
    assembler12 = (
        VectorAssembler()
        .setInputCols(["Curricular_units_1st_sem_without_evaluations"])
        .setOutputCol("vec_Curricular_units_1st_sem_without_evaluations")
    )
    assembler13 = (
        VectorAssembler()
        .setInputCols(["Curricular_units_2nd_sem_credited"])
        .setOutputCol("vec_Curricular_units_2nd_sem_credited")
    )
    assembler14 = (
        VectorAssembler()
        .setInputCols(["Curricular_units_2nd_sem_enrolled"])
        .setOutputCol("vec_Curricular_units_2nd_sem_enrolled")
    )
    assembler15 = (
        VectorAssembler()
        .setInputCols(["Curricular_units_2nd_sem_evaluations"])
        .setOutputCol("vec_Curricular_units_2nd_sem_evaluations")
    )
    assembler16 = (
        VectorAssembler()
        .setInputCols(["Curricular_units_2nd_sem_approved"])
        .setOutputCol("vec_Curricular_units_2nd_sem_approved")
    )
    assembler17 = (
        VectorAssembler()
        .setInputCols(["Curricular_units_2nd_sem_grade"])
        .setOutputCol("vec_Curricular_units_2nd_sem_grade")
    )
    assembler18 = (
        VectorAssembler()
        .setInputCols(["Curricular_units_2nd_sem_without_evaluations"])
        .setOutputCol("vec_Curricular_units_2nd_sem_without_evaluations")
    )

    df = assembler1.transform(df)
    df = assembler2.transform(df)
    df = assembler3.transform(df)
    df = assembler4.transform(df)
    df = assembler5.transform(df)
    df = assembler6.transform(df)
    df = assembler7.transform(df)
    df = assembler8.transform(df)
    df = assembler9.transform(df)
    df = assembler10.transform(df)
    df = assembler11.transform(df)
    df = assembler12.transform(df)
    df = assembler13.transform(df)
    df = assembler14.transform(df)
    df = assembler15.transform(df)
    df = assembler16.transform(df)
    df = assembler17.transform(df)
    df = assembler18.transform(df)

    scaler1 = MinMaxScaler(
        inputCol="vec_Previous_qualification_grade",
        outputCol="scaled_Previous_qualification_grade",
    )
    scaler2 = MinMaxScaler(
        inputCol="vec_Admission_grade", outputCol="scaled_Admission_grade"
    )
    scaler3 = MinMaxScaler(
        inputCol="vec_Unemployment_rate", outputCol="scaled_Unemployment_rate"
    )
    scaler4 = MinMaxScaler(
        inputCol="vec_Inflation_rate", outputCol="scaled_Inflation_rate"
    )
    scaler5 = MinMaxScaler(inputCol="vec_GDP", outputCol="scaled_GDP")
    scaler6 = MinMaxScaler(
        inputCol="vec_Age_at_enrollment", outputCol="scaled_Age_at_enrollment"
    )
    scaler7 = MinMaxScaler(
        inputCol="vec_Curricular_units_1st_sem_credited",
        outputCol="scaled_Curricular_units_1st_sem_credited",
    )
    scaler8 = MinMaxScaler(
        inputCol="vec_Curricular_units_1st_sem_enrolled",
        outputCol="scaled_Curricular_units_1st_sem_enrolled",
    )
    scaler9 = MinMaxScaler(
        inputCol="vec_Curricular_units_1st_sem_evaluations",
        outputCol="scaled_Curricular_units_1st_sem_evaluations",
    )
    scaler10 = MinMaxScaler(
        inputCol="vec_Curricular_units_1st_sem_approved",
        outputCol="scaled_Curricular_units_1st_sem_approved",
    )
    scaler11 = MinMaxScaler(
        inputCol="vec_Curricular_units_1st_sem_grade",
        outputCol="scaled_Curricular_units_1st_sem_grade",
    )
    scaler12 = MinMaxScaler(
        inputCol="vec_Curricular_units_1st_sem_without_evaluations",
        outputCol="scaled_Curricular_units_1st_sem_without_evaluations",
    )
    scaler13 = MinMaxScaler(
        inputCol="vec_Curricular_units_2nd_sem_credited",
        outputCol="scaled_Curricular_units_2nd_sem_credited",
    )
    scaler14 = MinMaxScaler(
        inputCol="vec_Curricular_units_2nd_sem_enrolled",
        outputCol="scaled_Curricular_units_2nd_sem_enrolled",
    )
    scaler15 = MinMaxScaler(
        inputCol="vec_Curricular_units_2nd_sem_evaluations",
        outputCol="scaled_Curricular_units_2nd_sem_evaluations",
    )
    scaler16 = MinMaxScaler(
        inputCol="vec_Curricular_units_2nd_sem_approved",
        outputCol="scaled_Curricular_units_2nd_sem_approved",
    )
    scaler17 = MinMaxScaler(
        inputCol="vec_Curricular_units_2nd_sem_grade",
        outputCol="scaled_Curricular_units_2nd_sem_grade",
    )
    scaler18 = MinMaxScaler(
        inputCol="vec_Curricular_units_2nd_sem_without_evaluations",
        outputCol="scaled_Curricular_units_2nd_sem_without_evaluations",
    )

    df = scaler1.fit(df).transform(df)
    df = scaler2.fit(df).transform(df)
    df = scaler3.fit(df).transform(df)
    df = scaler4.fit(df).transform(df)
    df = scaler5.fit(df).transform(df)
    df = scaler6.fit(df).transform(df)
    df = scaler7.fit(df).transform(df)
    df = scaler8.fit(df).transform(df)
    df = scaler9.fit(df).transform(df)
    df = scaler10.fit(df).transform(df)
    df = scaler11.fit(df).transform(df)
    df = scaler12.fit(df).transform(df)
    df = scaler13.fit(df).transform(df)
    df = scaler14.fit(df).transform(df)
    df = scaler15.fit(df).transform(df)
    df = scaler16.fit(df).transform(df)
    df = scaler17.fit(df).transform(df)
    df = scaler18.fit(df).transform(df)

    assembler = VectorAssembler(inputCols=inputCols, outputCol="features")
    df = assembler.transform(df)

    prediction = model.transform(df)

    label_mapping = {0.0: "Graduate", 1.0: "Dropout", 2.0: "Enrolled"}

    pandas_pred = prediction.toPandas()
    st.write("Prediction: " + label_mapping[pandas_pred.prediction.values[0]])
