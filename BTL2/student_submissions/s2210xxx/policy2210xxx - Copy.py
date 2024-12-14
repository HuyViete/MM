from policy import Policy
import numpy as np
from scipy.optimize import linprog
class Policy2210xxx(Policy):
    def __init__(self, policy_id=1):
        # assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy_id = 1
            
        elif policy_id == 2:
            self.policy_id = 2

        
    def _interpret_solution(self, solution, list_prods, stocks):
                    placements = []
                    num_prods = len(list_prods)
                    num_stocks = len(stocks)

                    for p_idx, prod in enumerate(list_prods):
                        prod_w, prod_h = prod["size"]
                        prod_quantity = prod["quantity"]

                        if prod_quantity > 0:
                            for s_idx in range(num_stocks):
                                index = p_idx * num_stocks + s_idx
                                if solution[index] > 0.5:  # Consider placing if the value is significant
                                    stock_w, stock_h = self._get_stock_size_(stocks[s_idx])

                                    # Check if product can fit within stock dimensions
                                    if stock_w >= prod_w and stock_h >= prod_h:
                                        # Try to place product on stock
                                        for x in range(stock_w - prod_w + 1):
                                            for y in range(stock_h - prod_h + 1):
                                                # Skip invalid positions
                                                if x + prod_w <= stock_w and y + prod_h <= stock_h:
                                                    # Now call can_place only for valid positions
                                                    if self._can_place_(stocks[s_idx], (x, y), (prod_w, prod_h)):
                                                        placement = {
                                                            "stock_idx": s_idx,
                                                            "size": (prod_w, prod_h),
                                                            "position": (x, y)
                                                        }
                                                        placements.append(placement)
                                                        prod_quantity -= 1
                                                        break
                                            if prod_quantity <= 0:
                                                break
                                if prod_quantity <= 0:
                                    break

                    return placements
    
    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            prods = observation["products"]
            sorted_prods = sorted(prods, key=lambda x: x["size"][0] * x["size"][1], reverse=True)
            sorted_stocks = sorted(enumerate(observation["stocks"]), key=lambda x: self._get_stock_size_(x[1])[0] * self._get_stock_size_(x[1])[1],)

            for prod in sorted_prods:
                if prod["quantity"] == 0:
                    continue
                
                prod_w, prod_h = prod["size"]

                # for i, stock in enumerate(observation["stocks"]):
                for i, stock in sorted_stocks:
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Fast placement: If the stock is empty, place in the top-left corner
                    if np.all(stock == -1):
                        return {
                            "stock_idx": i,
                            "size": (prod_w, prod_h),
                            "position": (0, 0),
                        }

                    # Evaluate placements and choose the best
                    best_fit = None
                    min_waste = float('inf')

                    for orientation in [(prod_w, prod_h), (prod_h, prod_w)]:
                        max_x = stock_w - orientation[0]
                        max_y = stock_h - orientation[1]

                        for x in range(max_x + 1):
                            for y in range(max_y + 1):
                                if self._can_place_(stock, (x, y), orientation):
                                    # Calculate waste for this placement
                                    used_area = orientation[0] * orientation[1]
                                    total_free_area = np.sum(stock == -1)
                                    waste = total_free_area - used_area

                                    # Update the best fit based on waste
                                    if waste < min_waste:
                                        min_waste = waste
                                        best_fit = {
                                            "stock_idx": i,
                                            "size": orientation,
                                            "position": (x, y),
                                            "waste": waste,
                                        }

                    # If a valid fit is found, use the best one
                    if best_fit:
                        return {
                            "stock_idx": best_fit["stock_idx"],
                            "size": best_fit["size"],
                            "position": best_fit["position"],
                        }

            # If no placement is found
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        elif self.policy_id==2: 
                
                list_prods = observation["products"]
                stocks = observation["stocks"]

                num_prods = len(list_prods)
                num_stocks = len(stocks)

                # Flatten the stocks into a single LP representation
                stock_dims = [self._get_stock_size_(stock) for stock in stocks]

                # LP formulation
                c = []  # Coefficients for the objective function (minimize waste)
                A_eq = []  # Equality constraints (fulfill product demands)
                b_eq = []  # RHS for equality constraints

                # Build LP problem
                for p_idx, prod in enumerate(list_prods):
                    prod_w, prod_h = prod["size"]
                    prod_quantity = prod["quantity"]

                    if prod_quantity > 0:
                        row = [0] * (num_prods * num_stocks)  # Initialize the row for A_eq

                        # Track if any possible placements were found
                        any_possible_placement = False

                        for s_idx, (stock_w, stock_h) in enumerate(stock_dims):
                            index = p_idx * num_stocks + s_idx

                            # Check if product can fit in stock
                            if stock_w >= prod_w and stock_h >= prod_h:
                                # Find a possible placement
                                placement_found = False
                                for x in range(stock_w - prod_w + 1):
                                    for y in range(stock_h - prod_h + 1):
                                        if self._can_place_(stocks[s_idx], (x, y), (prod_w, prod_h)):
                                            row[index] = 1
                                            wasted_space = stock_w * stock_h - prod_w * prod_h
                                            c.append(wasted_space)
                                            placement_found = True
                                            any_possible_placement = True
                                            break
                                    if placement_found:
                                        break
                                
                                if not placement_found:
                                    row[index] = 0
                                    c.append(1000000)
                            else:
                                row[index] = 0
                                c.append(1000000)

                        # Only add constraint if at least one placement is possible
                        if any_possible_placement:
                            A_eq.append(row)
                            b_eq.append(prod_quantity)

                # If no constraints were added, return default action
                if not A_eq:
                    return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

                # Convert to numpy arrays
                c = np.array(c)
                A_eq = np.array(A_eq)
                b_eq = np.array(b_eq)

                # Pad c to match A_eq columns if needed
                if len(c) < A_eq.shape[1]:
                    c = np.pad(c, (0, A_eq.shape[1] - len(c)), mode='constant', constant_values=1000000)

                # Solve the linear program
                res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method="highs", options={"maxiter": 10000, "tol": 1e-6})

                if res.success:
                    solution = res.x  # Fractional solutions for placement
                    # Get placements from solution
                    placements = self._interpret_solution(solution, list_prods, stocks)

                    if placements:
                        # Return the first valid placement
                        return placements[0]

                # Fallback if no placement is found
                return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

                

                # Student code here
                # You can add more functions if needed
