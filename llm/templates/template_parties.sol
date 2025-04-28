contract PartiesTemplate {
    // Landlord Information
    string public landlordName;
    string public landlordMailingAddress;
    // Consider using address type for on-chain identity if applicable
    // address public landlordAddress;

    // Tenant Information
    string public tenantName;
    string public tenantMailingAddress;
    // address public tenantAddress;

    // Occupant Information (Could be an array if multiple occupants)
    string[] public occupantNames;

    // Agreement Date
    uint256 public agreementDate; // Unix timestamp
}