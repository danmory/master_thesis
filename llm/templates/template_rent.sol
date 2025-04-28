contract RentTemplate {
    enum RentPaymentType { FixedAmount, MonthlyAmount }

    RentPaymentType public rentPaymentType;
    uint256 public totalRentAmount; // For FixedAmount
    uint256 public monthlyRentAmount; // For MonthlyAmount
    uint256 public rentDueDateDay; // Day of the month rent is due (for MonthlyAmount)
    string public paymentInstructions; // For MonthlyAmount
}