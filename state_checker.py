import pandas as pd

def validate_data_counts(csv_path):
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Define expected counts based on failure and verbal numbers
    def expected_count(failure, verbal):
        if failure == 0:
            return 11
        elif failure == 1:
            return 13
        elif failure == 2:
            return 21 if verbal == 2 else 11
        elif failure == 3:
            return 21 if verbal == 2 else 11
        # elif failure in (2, 3):
        #     return 21 if verbal == 2 else 11
        else:
            return None

    # Grouping keys
    group_cols = ['Puzzle_number', 'object_number', 'failure_number', 'Verbal_number']

    # Collect any mismatches
    mismatches = []

    # Iterate over each group and compare actual vs. expected count
    for name, group in df.groupby(group_cols):
        puzzle, obj, failure, verbal = name
        actual = len(group)
        expected = expected_count(failure, verbal)

        if expected is None:
            mismatches.append((name, actual, 'Unexpected failure number'))
        elif actual != expected:
            mismatches.append((name, actual, expected))

    # Report results
    if not mismatches:
        print("All group counts match expected values.")
    else:
        print("Found mismatches in data counts:")
        for (name, actual, expected) in mismatches:
            puzzle, obj, failure, verbal = name
            print(
                f"Puzzle {puzzle}, Object {obj}, Failure {failure}, "
                f"Verbal {verbal}: actual={actual}, expected={expected}"
            )

# Example usage:
for i in range(1,55):
    if i==3 or i == 9 or i==47:
        continue
    print(i)
    validate_data_counts(f'/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data/Participant {i}/ee_state_data.csv')

