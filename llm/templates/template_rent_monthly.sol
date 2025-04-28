contract RentTemplateMonthly {
    uint256 public monthlyRentAmount = 1000;
    uint256 public rentDueDateDay = 1;
    
    uint256 public lastRentPaymentMonth;
    
    function payMonthlyRent() external payable {
        require(msg.value == monthlyRentAmount, "Incorrect rent amount");
        uint256 currentMonth =  block.timestamp / 30 days + 1;
        require(lastRentPaymentMonth != currentMonth, "Already paid");
        lastRentPaymentMonth = currentMonth;
    }
}