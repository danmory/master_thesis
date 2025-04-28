contract RentTemplateFixed {
    uint256 public totalRentAmount = 12000;
    
    bool public rentPaid;
    
    function payFixedRent() external payable {
        require(msg.value == totalRentAmount, "Incorrect rent amount");
        require(!rentPaid, "Already paid");
        rentPaid = true;
    }
}