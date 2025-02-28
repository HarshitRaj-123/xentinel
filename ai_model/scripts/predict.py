import pandas as pd

def predict_crime_pattern(text, location, time, severity):
    """
    Predict crime pattern based on input data.
    Replace this with your actual ML model.
    """
    # Dummy prediction logic
    if "theft" in text.lower():
        return "Theft"
    elif "assault" in text.lower():
        return "Assault"
    else:
        return "Other"

def analyze_crime_data(crime_data):
    """
    Analyze crime data to generate statistics.
    """
    # Total number of crimes
    total_crimes = len(crime_data)
    
    # Most prominent crime type
    crime_type_counts = crime_data["crime_type"].value_counts().to_dict()
    most_prominent_crime_type = max(crime_type_counts, key=crime_type_counts.get)
    
    # Most prominent crime sub-type
    if "crime_subtype" in crime_data.columns:
        crime_subtype_counts = crime_data["crime_subtype"].value_counts().to_dict()
        most_prominent_crime_subtype = max(crime_subtype_counts, key=crime_subtype_counts.get)
    else:
        most_prominent_crime_subtype = "N/A"
    
    # Seriousness of cases
    if "severity" in crime_data.columns:
        seriousness_counts = crime_data["severity"].value_counts().to_dict()
    else:
        seriousness_counts = {"N/A": total_crimes}
    
    # Total cases in the database
    total_cases = total_crimes  # Same as total_crimes
    
    # Return statistics
    return {
        "total_crimes": total_crimes,
        "most_prominent_crime_type": most_prominent_crime_type,
        "most_prominent_crime_subtype": most_prominent_crime_subtype,
        "seriousness_of_cases": seriousness_counts,
        "total_cases_in_database": total_cases,
    }