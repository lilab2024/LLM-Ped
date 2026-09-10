import pandas as pd
import json


csv_path = "Cascaded_Retrieval-Augmented_Fine-Tuning/dataset/csv-data/train.csv"   # csv data
df = pd.read_csv(csv_path)

#============INPUT_PROMPT==========

pedestrian_type_map = {
    "A": "person on foot",
    "B": "person on bike",
    "C": "person on vehicle",
    "D": "person walking bike",
    "E": "mixed types",
    "F": "other",
    "G": "person with a dog",
    "H": "person with a stroller or small child"
}

opposite_direction_yield_map = {
    0: "absent",
    1: "opposite direction vehicle yields",
    2: "opposite direction vehicle does not yield"
}

following_vehicle_map = {
    0: "absent",
    1: "following vehicle exists"
}

bike_lane_map = {
    0: "bike lanes absent",
    1: "bike lanes present"
}

weather_map = {
    0: "no precipitation",
    1: "rain",
    2: "snow"
}

signage_map = {
    0: "no signage",
    1: "signed crosswalk"
}
markings_map = {
    "U": "unmarked",
    "C": "continental",
    "S": "standard"
}
presence_map = {
    0: "absent",
    1: "present"
}
on_street_parking_map = {
    0: "none",
    1: "one-sided",
    2: "two-sided"
}
tree_cover_map = {
    0: "no tree cover",
    1: "very low tree cover",
    2: "low tree cover",
    3: "moderate tree cover",
    4: "high tree cover"
}

lighting_map = {
    0: "no lighting",
    1: "lighting present"
}
road_surface_map = {
    0: "dry",
    1: "wet"
}


#=====INSTRUCTION_PROMPT==================

instruction_text="""
You are a helpful assistant designed to predict whether a driver will yield to a pedestrian at unsignalized intersections. Please predict whether the driver will yield to pedestrian(s).
- 1 = vehicle yields to pedestrian(s).
- 0 = vehicle failed to yield.
Output: The answer is 0 or The answer is 1.
"""

input_prompt_templete=""""
1. Pedestrian Movement and Path

    Location ID: {Location ID}
    The pedestrian arrived at the curb and showed an intention to cross at {Time Showed Intent} (HHMMSS),
    and began crossing at {Time Started Crossing} (HHMMSS).
    Pedestrian group size: {Number of Pedestrians}
    Pedestrian type: {Pedestrian Type}
    Pedestrian origin corner: {Pedestrian Origin}
    Pedestrian destination corner: {Pedestrian Destination}
    Intersection corner definitions:
        - Corners A and C are on the same side of the roadway.
        - Corners B and D are on the opposite side.
        - Corner A is directly opposite corner B.
        - Corner C is directly opposite corner D.

2. Vehicle State and Traffic context

    Approaching vehicle speed: {Vehicle Speed} mph
    Posted speed limit: {Posted Speed} mph
    Opposite-direction vehicle yield condition: {Opposite Direction Yield}
    Presence of a following vehicle: {Following Vehicle}

    PAWS score(walkability score): {PAWS Score}.
    Number of bus stops within one block: {Number of bus stops}.
    Minor road Annual Average Daily Traffic : {Minor AADT}.
    Major road Annual Average Daily Traffic : {Major AADT}.

3. Roadway Geometry and Crossing Facilities

    Number of lanes on the major road: {Number Lanes Main}
    Total pedestrian crossing width: {Crossing Width (Major)} feet
    Bike lanes: {Bike Lane(s)}
    Crosswalk signage: {Signage}
    Crosswalk markings: {Markings}
    Street lighting: {Lighting}

4. Environmental Conditions
    
    Weather: {Weather}
    Road surface condition: {Road Surface}
    Tree cover: {Tree Cover}

5. Surrounding Land Use 

    Single-family residences present: {Presence of Single Family}
    Apartments present: {Presence of Apartments}
    Commercial areas present: {Presence of Commercial}
    Restaurants or bars present: {Presence of Restaurants/Bars}
    Parking lots present: {Presence of Parking Lots}
    Gas stations or convenience stores present: {Presence of Gas Station/Convenient Store}
    On-street parking present: {Presence of on street parking}
    Distance to nearest park: {Dist to Nearest Park} miles
    Distance to nearest school: {Dist to Nearest School} miles

Note: "nan" indicates missing data."""


alpaca_data = []

for _, row in df.iterrows():
    row_mapping = {
        "Location ID": row['Location ID'],
        "Time Showed Intent": row['Time Showed Intent'],
        "Time Started Crossing": row['Time Started Crossing'],
        "Number of Pedestrians": row['Number of Pedestrians'],
        "Pedestrian Origin": row['Pedestrian Origin'],
        "Pedestrian Destination": row['Pedestrian Destination'],
        "Pedestrian Type": pedestrian_type_map[row['Pedestrian Type']],
        "Vehicle Speed": row['Vehicle Speed'],
        "Opposite Direction Yield": opposite_direction_yield_map[row['Opposite Direction Yield']],
        "Following Vehicle": following_vehicle_map[row['Following Vehicle']],
        "Posted Speed": row['Posted Speed'],
        "Number Lanes Main": row['Number Lanes Main'],
        "Crossing Width (Major)": row['Crossing Width (Major)'],
        "Bike Lane(s)": bike_lane_map[row['Bike Lane(s)']],
        "Weather": weather_map[row['Weather']],
        "Signage": signage_map[row['Signage']],
        "Markings": markings_map[row['Markings']],
        "Presence of Single Family": presence_map[row['Presence of Single Family']],
        "Presence of Apartments": presence_map[row['Presence of Apartments']],
        "Presence of Commercial": presence_map[row['Presence of Commercial']],
        "Presence of Gas Station/Convenient Store": presence_map[row['Presence of Gas Station/Convenient Store']],
        "Presence of Restaurants/Bars": presence_map[row['Presence of Restaurants/Bars']],
        "Presence of Parking Lots": presence_map[row['Presence of Parking Lots']],
        "Dist to Nearest Park": row['Dist to Nearest Park'],
        "Dist to Nearest School": row['Dist to Nearest School'],
        "Presence of on street parking": on_street_parking_map[row['Presence of on street parking']],
        "PAWS Score": row['PAWS Score'],
        "Tree Cover": tree_cover_map[row['Tree Cover']],
        "Lighting": lighting_map[row['lighting']],
        "Road Surface": road_surface_map[row['road surface']],
        "Number of bus stops": row['Number of bus stops'],
        "Minor AADT": row['Minor AADT'],
        "Major AADT": row['Major AADT']
    }
    

    input_text=input_prompt_templete.format(**row_mapping)

    # print(instruction_text)
    # print(input_text)

    alpaca_sample = {
        "instruction": instruction_text.strip(),
        "input": input_text.strip(),
        "output":  "<think>\n</think>\n The answer is: "+str(row["target"])
    }

    alpaca_data.append(alpaca_sample)



with open("Cascaded_Retrieval-Augmented_Fine-Tuning/dataset/train_dataset.json", "w", encoding="utf-8") as f:
    json.dump(alpaca_data, f, ensure_ascii=False, indent=2)


print("prompt design finishing")
