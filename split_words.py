import re

def split_camel_case(word):
    return re.sub('([a-z])([A-Z])', r'\1 \2', word).lower()

# Test the function
print(split_camel_case("activityCategoryName"))  # activity category name
print(split_camel_case("activityCategoryDescText"))  # activity category desc text
