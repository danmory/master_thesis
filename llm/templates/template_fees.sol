contract FeesTemplate {
    bool public cleaningFeeRequired = true;
    uint256 public cleaningFeeAmount = 5;

    bool public taxesRequired = true;
    uint256 public taxesAmount = 10;

    bool public otherFeeRequired = true;
    string public otherFeeDescription = "parking";
    uint256 public otherFeeAmount = 14;
}