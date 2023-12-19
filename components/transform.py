from decimal import Decimal
import re

def decimal_to_float(obj):
    if isinstance(obj, Decimal):
        return round(float(obj),2)
    elif isinstance(obj, list):
        return [decimal_to_float(item) for item in obj]
    elif isinstance(obj, tuple):
        return [decimal_to_float(item) for item in obj]
    else :
        return obj

def replace_aggregation_functions(query, new_table_name):
    # Regular expression to rename table
    pattern = re.compile(r'\bFROM\s+(\w+)')
    query = pattern.sub(f'FROM {new_table_name}', query)

    aggregation_functions = ['AVG', 'COUNT', 'SUM', 'MIN', 'MAX', 'AVG']

    # Regular expression pattern to match aggregation functions followed by a space and a word
    pattern = re.compile(r'(' + '|'.join(aggregation_functions) + r')\s+(\w+)')

    # Check if any aggregation function is found
    if pattern.search(query):
        return True, pattern.sub(r'\1(\2)', query)
    else:
        # If no aggregation function is found, replace all column names with '*'
        return False, re.sub(r'SELECT\s+(.*?)(?:\s+FROM|$)', r'SELECT * FROM', query,flags=re.IGNORECASE)