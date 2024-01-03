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



def fetch_all_rows_and_format(cursor, schema):

    # Get a list of all tables in the database
    cursor.execute(f"SELECT table_name FROM information_schema.tables WHERE table_schema='{schema}';")
    tables = [a[0] for a in cursor.fetchall() if 'knowled' not in a[0] and 'pg_' not in a[0] and '_embedd' not in a[0] and '_cache' not in a[0]]

    # Initialize the result string
    result_string = ""

    # Iterate through each table
    for table_name in tables:

        print(table_name)

        # Get the column names for the table
        cursor.execute(sql.SQL("SELECT column_name FROM information_schema.columns WHERE table_name = %s;"), [table_name])
        columns = cursor.fetchall()
        column_names = [column[0] for column in columns]

        # Fetch all rows from the table
        cursor.execute(sql.SQL("SELECT * FROM {}").format(sql.Identifier(table_name)))
        rows = cursor.fetchall()

        # Format the result string for the table
        result_string += f"\n{table_name} : \n"
        result_string += " | ".join(column_names) + "\n"
        for row in rows:
            result_string += " | ".join(map(str, row)) + "\n"
    print(result_string)
    return result_string