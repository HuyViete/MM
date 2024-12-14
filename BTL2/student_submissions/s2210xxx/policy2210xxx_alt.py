from policy import Policy
import numpy as np


class Policy2210xxx(Policy):
    def __init__(self, policy_id=1):
        # assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy_id = 1
            
        elif policy_id == 2:
            self.policy_id = 2

        elif policy_id == 3:
            self.policy_id = 3

        elif policy_id == 4:
            self.policy_id = 4

    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            stocks = observation.get("stocks", [])
            products = observation.get("products", [])

            if not stocks or not products:
                print("Dữ liệu stocks hoặc products trống!")
                return None

            # Sort the products in decreasing order
            sorted_products = sorted(products, key=lambda p: p["size"][0] * p["size"][1], reverse=True)

            # Iterate through each product
            for product in sorted_products:
                size = product["size"]
                quantity = product["quantity"]

                if quantity > 0:
                    for stock_idx, stock in enumerate(stocks):
                        stock_array = np.array(stock)

                        # Find the best position to cut
                        best_fit = None
                        for x in range(stock_array.shape[0] - size[0] + 1):
                            for y in range(stock_array.shape[1] - size[1] + 1):
                                sub_stock = stock_array[x:x + size[0], y:y + size[1]]

                                if np.all(sub_stock == -1):
                                    # Evalueate the area
                                    waste = (
                                        stock_array.shape[0] * stock_array.shape[1] -
                                        size[0] * size[1]
                                    )
                                    if best_fit is None or waste < best_fit["waste"]:
                                        best_fit = {"x": x, "y": y, "waste": waste}

                        if best_fit:
                            return {
                                "stock_idx": stock_idx,
                                "size": size,
                                "position": (best_fit["x"], best_fit["y"]),
                            }

            return None

        elif self.policy_id == 2:
            """
            Implements the Split Fit Algorithm:
            - Divide products into two groups based on width vs height.
            - Place each group into the stock separately.
            """
            list_prods = observation["products"]
            stocks = observation["stocks"]

            prod_size = [0, 0]
            stock_idx = -1
            pos_x, pos_y = 0, 0

            # Divide products into two groups
            group_1 = [prod for prod in list_prods if prod["quantity"] > 0 and prod["size"][0] > prod["size"][1]]  # Wide products
            group_2 = [prod for prod in list_prods if prod["quantity"] > 0 and prod["size"][0] <= prod["size"][1]] # Tall products

            # Process groups sequentially
            for group in [group_1, group_2]:
                for prod in group:
                    prod_size = prod["size"]

                    # Try to place this product in one of the stocks
                    for i, stock in enumerate(stocks):
                        stock_w, stock_h = self._get_stock_size_(stock)
                        prod_w, prod_h = prod_size

                        # Check placement without rotation
                        if stock_w >= prod_w and stock_h >= prod_h:
                            pos_x, pos_y = None, None
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), prod_size):
                                        pos_x, pos_y = x, y
                                        break
                            if pos_x is not None and pos_y is not None:
                                stock_idx = i
                                break

                        # Check placement with rotation
                        if stock_w >= prod_h and stock_h >= prod_w:
                            pos_x, pos_y = None, None
                            for x in range(stock_w - prod_h + 1):
                                for y in range(stock_h - prod_w + 1):
                                    if self._can_place_(stock, (x, y), prod_size[::-1]):
                                        prod_size = prod_size[::-1]
                                        pos_x, pos_y = x, y
                                        break
                            if pos_x is not None and pos_y is not None:
                                stock_idx = i
                                break

                    # If placed, return the action
                    if pos_x is not None and pos_y is not None:
                        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

            return None
        
        elif self.policy_id == 3:
            list_prods = observation["products"]
            sorted_prods = sorted(list_prods, key=lambda x: x["size"][0] * x["size"][1], reverse=True)

            for prod in sorted_prods:
                if prod["quantity"] > 0:
                    size = prod["size"]
                    prod_w, prod_h = size

                    for i, stock in enumerate(observation["stocks"]):
                        placed = False
                        stock_w, stock_h = self._get_stock_size_(stock)
                        fit = None
                        waste = np.sum(stock == -1) - prod_w * prod_h

                        if waste >= 0:
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), size):
                                        fit = {"x": x, "y": y, "size": size, "waste": waste}
                                        placed = True
                                        break

                                if placed:
                                    break

                            if fit is None:
                                for x in range(stock_w - prod_h + 1):
                                    for y in range(stock_h - prod_w + 1):
                                        if self._can_place_(stock, (x, y), size[::-1]):
                                            fit = {"x": x, "y": y, "size": size[::-1], "waste": waste}
                                            placed = True
                                            break

                                    if placed:
                                        break

                        if fit:
                            return {"stock_idx": i, "size": fit["size"], "position": (fit["x"], fit["y"])}

            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        elif self.policy_id == 4:
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

            return None

    # Student code here
    # You can add more functions if needed
