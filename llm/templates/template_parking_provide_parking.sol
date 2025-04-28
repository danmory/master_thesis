contract ParkingProvideParkingTemplate {
    bool public parkingProvided = true;
    uint256 public numberOfParkingSpaces = 1;
    uint256 public parkingFeeAmount = 100;
    enum ParkingFeePaymentTiming { AtExecution, Monthly }
    ParkingFeePaymentTiming public parkingFeePaymentTiming;
    string public parkingAreaDescription = "Underground garage space.";
}