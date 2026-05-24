def build_prompt_t1(row):
    """
    Builds the prompt for Pronominal Addressing tasks (T1).
    """
    situation = row.get("Situation", "").strip()
    
    if situation:
        return f"""
প্রেক্ষাপট:
{situation}

প্রশ্ন:
{row['Question']}

এক কথায় উত্তর দিনঃ আপনি/তুমি/তুই। কোন ব্যাখ্যা যুক্ত করবেন না।
""".strip()
    else:
        return f"""
প্রশ্ন:
{row['Question']}

এক কথায় উত্তর দিনঃ আপনি/তুমি/তুই। কোন ব্যাখ্যা যুক্ত করবেন না।
""".strip()


def build_prompt_t2(row):
    """
    Builds the prompt for Nominal Addressing / Custom tasks (T2).
    """
    situation = row.get("Situation", "").strip()

    options = "\n".join([
        f"1. {row['OptionA']}",
        f"2. {row['OptionB']}",
        f"3. {row['OptionC']}",
        f"4. {row['OptionD']}",
    ])

    instruction = "কোন অতিরিক্ত শব্দ বা ব্যাখ্যা যোগ না করে, শুধু সঠিক উত্তরের নম্বরটি লিখুন (1/2/3/4)।"

    if situation:
        return f"""
প্রেক্ষাপট:
{situation}

প্রশ্ন:
{row['Question']}

বিকল্পসমূহ:
{options}

{instruction}
""".strip()
    else:
        return f"""
প্রশ্ন:
{row['Question']}

বিকল্পসমূহ:
{options}

{instruction}
""".strip()