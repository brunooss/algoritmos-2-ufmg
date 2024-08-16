from ucimlrepo import fetch_ucirepo

estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = fetch_ucirepo(
    id=544)

X = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.features
y = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.targets
print(X.values)
print(y.values)
