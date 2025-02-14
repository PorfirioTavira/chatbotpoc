import random

# Set how many expressions you want:
NUM_EXPRESSIONS = 30000

def main():
    # Define the textual representation of operators
    operator_text = {
        '+': "plus",
        '-': "minus",
        '*': "times",
        '/': "divided by"
    }
    
    # List of operators from which to choose
    operators = list(operator_text.keys())
    
    with open("../arithmatic.txt", "w") as f:
        for _ in range(NUM_EXPRESSIONS):
            # Randomly pick an operator
            op = random.choice(operators)
            
            # Generate two random real numbers
            a = random.uniform(-10, 10)
            b = random.uniform(-10, 10)
            
            # If operator is division, ensure the denominator isn't zero
            if op == '/':
                while b == 0:
                    b = random.uniform(-10, 10)
            
            # Compute the result based on the operator
            if op == '+':
                result = a + b
            elif op == '-':
                result = a - b
            elif op == '*':
                result = a * b
            else:  # op == '/'
                result = a / b
            
            # Write the expression to the file in a textual format
            # For neatness, format numbers to two decimal places
            f.write(f"{a} {op} {b} = {result}\n")

if __name__ == "__main__":
    main()
