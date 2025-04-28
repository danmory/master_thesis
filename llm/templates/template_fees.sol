contract FeesTemplate {
    bool public cleaningFeeRequired = true;
    uint256 public cleaningFeeAmount = 5;

    bool public taxesRequired = true;
    uint256 public taxesAmount = 10;

    bool public otherFeeRequired = true;
    string public otherFeeDescription = "parking";
    uint256 public otherFeeAmount = 14;
    
    bool public cleaningFeePaid;
    bool public taxesPaid;
    bool public otherFeePaid;
    
    function payCleaningFee() external payable {
        require(cleaningFeeRequired, "Cleaning fee not required");
        require(msg.value == cleaningFeeAmount, "Incorrect cleaning fee amount");
        require(!cleaningFeePaid, "Already paid");
        cleaningFeePaid = true;
    }
    
    function payTaxes() external payable {
        require(taxesRequired, "Taxes not required");
        require(msg.value == taxesAmount, "Incorrect taxes amount");
        require(!taxesPaid, "Already paid");
        taxesPaid = true;
    }
    
    function payOtherFee() external payable {
        require(otherFeeRequired, "Other fee not required");
        require(msg.value == otherFeeAmount, "Incorrect other fee amount");
        require(!otherFeePaid, "Already paid");
        otherFeePaid = true;
}