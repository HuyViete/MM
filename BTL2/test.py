from scipy.optimize import linprog

# Example data
items = [
    {'length': 4, 'width': 3, 'demand': 2},  # Item 1
    {'length': 2, 'width': 2, 'demand': 3}   # Item 2
]
stocks = [
    {'length': 10, 'width': 10},  # Stock 1
    {'length': 12, 'width': 8}   # Stock 2
]

# Define variables
num_items = len(items)
num_stocks = len(stocks)
num_variables = num_items * num_stocks

# Objective function: minimize waste (w_j for each stock)
c = [0] * num_variables + [1] * num_stocks  # x_ij coefficients + w_j coefficients

# Constraints
A_eq = []
b_eq = []

# Demand constraints: sum of x_ij for each item = demand
for i, item in enumerate(items):
    row = [0] * num_variables + [0] * num_stocks
    for j in range(num_stocks):
        row[i * num_stocks + j] = 1
    A_eq.append(row)
    b_eq.append(item['demand'])

A_ub = []
b_ub = []

# Stock space constraints: handle rotation
for j, stock in enumerate(stocks):
    stock_area = stock['length'] * stock['width']
    for i, item in enumerate(items):
        item_area = item['length'] * item['width']
        row = [0] * num_variables + [0] * num_stocks
        row[i * num_stocks + j] = item_area  # Add area constraint
        row[num_variables + j] = -1  # Subtract waste
        A_ub.append(row)
        b_ub.append(stock_area)

# Non-negativity constraints are implicit in linprog

# Solve the LP
result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=(0, None))

# Process the results
if result.success:
    print("Optimal solution found!")
    print("x_ij values (cuts):", result.x[:num_variables])
    print("Waste per stock:", result.x[num_variables:])
else:
    print("No optimal solution found.")