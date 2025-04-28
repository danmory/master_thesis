contract PremisesTemplate {
    string public mailingAddress;
    string public residenceType; // e.g., "Apartment", "House", "Condo", or specific type
    uint256 public numberOfBedrooms;
    uint256 public numberOfBathrooms;
    string public otherDescription; // Description of any other relevant features

    // --- Premises Details ---
    // mailingAddress: The full address of the property being leased.
    // residenceType: The type of dwelling.
    // numberOfBedrooms: Count of bedrooms.
    // numberOfBathrooms: Count of bathrooms.
    // otherDescription: Any additional description provided.
}