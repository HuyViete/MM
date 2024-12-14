from policy import Policy
import numpy as np


class Policy2352429(Policy):
    def __init__(self):
        # Student code here
        """
        Initialize the policy with necessary parameters.
        """
        super().__init__()

    def get_action(self, observation, info):
        # Student code here
        """
        Select an action based on the current observation and additional information.

        Args:
            observation (dict): Current observation state (stocks, products).
            info (dict): Additional information from the environment.

        Returns:
            dict | None: A valid action or None if no suitable action is found.
        """
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

        # Student code here
        # You can add more functions if needed