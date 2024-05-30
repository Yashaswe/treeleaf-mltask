import requests
import json

def get_user_input():
    user_input = {}
    while True:
        try:
            user_input['Age'] = float(input("Enter Age: "))
            gender = input("Enter your gender (M/F/O): ").strip().upper()
            if gender == 'M':
                user_input['Gender'] = 0
            elif gender == 'F':
                user_input['Gender'] = 1
            elif gender == 'O':
                user_input['Gender'] = 2
            else:
                raise ValueError("Invalid gender input. Please enter 'M', 'F', or 'O'.")

            user_input['Experience'] = float(input("Enter Experience: "))
            user_input['Income'] = float(input("Enter Income: "))
            user_input['Family'] = int(input("Enter Family size: "))
            user_input['CCAvg'] = float(input("Enter CCAvg: "))
            home_ownership = input("Enter Home Ownership (Home Mortgage/Home Owner/Rent): ").strip().title()
            if home_ownership == 'Home Mortgage':
                user_input['Home Ownership'] = 0
            elif home_ownership == 'Home Owner':
                user_input['Home Ownership'] = 1
            elif home_ownership == 'Rent':
                user_input['Home Ownership'] = 3
            else:
                raise ValueError("Invalid home ownership input. Please enter 'Home Mortgage', 'Home Owner', or 'Rent'.")

            user_input['Education'] = int(input("Enter Education level (1-3): "))
            user_input['Mortgage'] = float(input("Enter Mortgage: "))
            user_input['Securities Account'] = int(input("Enter Securities Account (0 or 1): "))
            user_input['CD Account'] = int(input("Enter CD Account (0 or 1): "))
            user_input['Online'] = int(input("Enter Online (0 or 1): "))
            user_input['CreditCard'] = int(input("Enter CreditCard (0 or 1): "))
            break
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")
    return user_input

def main():
    url = 'http://localhost:5000/predict'
    
    while True:
        user_input = get_user_input()
        response = requests.post(url, json=user_input)
        result = response.json()
        
        if 'prediction' in result:
            print(f"Loan Prediction: {result['prediction']}")
        else:
            print(f"Error: {', '.join(result['error'])}")
        
        cont = input("Do you want to enter another record? (yes/no): ")
        if cont.lower() != 'yes':
            break

if __name__ == "__main__":
    main()
