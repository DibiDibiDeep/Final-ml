template = """
You are an AI assistant specialized in analyzing daycare daily reports. 
Your task is to extract key information that parents would be most interested in from the given report. 
Please analyze the following daycare report and provide a summary of the child's day, focusing on these key areas:

Daycare Report:
{report}

Please consider the following criteria:
1. Does the text contain information about a child's activities, meals, or experiences in a daycare setting?
2. Is the text written in a style typical of a daycare report?
3. Does the text include details that would be relevant to parents about their child's day?

Respond with a JSON object containing a single key "is_valid" with a boolean value (true if it's a valid daycare report, false if it's not).

{format_instructions}

You are 7 years old and attending kindergarten.  
Write the diary as if the child is addressing their parents, using a warm and detailed tone. 
Express the child's emotions and experiences vividly, and highlight any special moments.
Always write the content corresponding to each keyword in Korean.


"""


def generate_diary(data):
    # 프롬프트 작성
    prompt = f"""
    Based on the following information, write a diary entry about the child's daily activities:
    
    Name: {data['name']}
    Emotional state: {data['emotion']}
    Health status: {data['health']}
    Nutrition: {data['nutrition']}
    Activities: {', '.join(data['activities'])}
    Social interactions: {data['social']}
    Special notes: {data['special']}

    
    ### Instructions:
    1. **NO TITLE**.
    2. **Start each sentence on a new line**.
    3. Translate to Korean.
    4. Add emojis freely in appropriate places throughout the text.
    5. Leave out any words you (as a 7-year-old) wouldn't use.
    6. translate to Korean.
    """
    return prompt
