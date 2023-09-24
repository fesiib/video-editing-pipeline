
def main():
    file_name = "cost.txt"
    total_cost = 0
    with open(file_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split("$")
            if len(parts) < 2:
                continue
            cost = float(parts[1])
            print("Cost: {}".format(parts[1]))
            total_cost += float(cost)
    print("Total cost: ${:.2f}".format(total_cost))


if __name__ == "__main__":
    main()