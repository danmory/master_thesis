contract SecurityDepositTemplate {
    bool public securityDepositRequired = true;
    uint256 public securityDepositAmount = 100; // 100 $
    
    bool public securityDepositPaid;
    
    function paySecurityDeposit() external payable {
        require(securityDepositRequired, "Security deposit not required");
        require(msg.value == securityDepositAmount, "Incorrect security deposit amount");
        require(!securityDepositPaid, "Already paid");
        securityDepositPaid  = true;
    }
}
