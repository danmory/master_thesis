contract PetsTemplate {
    bool public petsAllowed = true;
    uint256 public maxNumberOfPets = 2;
    uint256 public maxWeightPerPetLbs = 10;
    uint256 public petDepositAmount = 50; // $50
    bool public petDepositRefundable = true;
}