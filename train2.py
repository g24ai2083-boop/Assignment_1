from sklearn.kernel_ridge import KernelRidge
from misc import load_data, preprocess_data, split_data, train_model, evaluate_model

def run_kernelridge_workflow():
    # 1. Data Loading
    df = load_data()

    # 2. Data Preprocessing (uses the same generic function)
    X, y = preprocess_data(df)

    # 3. Data Splitting (uses the same generic function)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 4. Model Training
    # Initialize KernelRidge model (you may tune the alpha/kernel if desired)
    model = KernelRidge(alpha=1.0, kernel='linear')
    trained_model = train_model(model, X_train, y_train)

    # 5. Model Evaluation
    mse = evaluate_model(trained_model, X_test, y_test)

    # Display the average MSE score [cite: 25]
    print(f"KernelRidge Test MSE: {mse:.4f}")
    return mse

if __name__ == "__main__":
    run_kernelridge_workflow()