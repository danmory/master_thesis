// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ServiceAgreement {
    address public serviceProvider;
    address public client;
    string public serviceDescription;
    uint256 public serviceFee;
    uint256 public paymentFrequency;
    uint256 public agreementEndDate;
    mapping(uint => bool) payments;
    
    enum ServiceStatus { Pending, Active, Completed, Terminated }
    ServiceStatus public status;
    
    event PaymentMade(address indexed sender, uint256 amount);
    event ServiceDelivered(address indexed provider, string description);
    event AgreementTerminated(address indexed sender);
    event PaymentVerified(address indexed sender, uint256 paymentNumber);
    
    constructor(
        address _serviceProvider,
        address _client,
        string memory _serviceDescription,
        uint256 _serviceFee,
        uint256 _paymentFrequency,
        uint256 _agreementEndDate
    ) {
        serviceProvider = _serviceProvider;
        client = _client;
        serviceDescription = _serviceDescription;
        serviceFee = _serviceFee;
        paymentFrequency = _paymentFrequency;
        agreementEndDate = _agreementEndDate;
        status = ServiceStatus.Pending;
    }
    
    function makePayment() public {
        require(msg.sender == client, "Only the client can make payments.");
        require(block.timestamp < agreementEndDate, "Agreement has ended.");
        require(status == ServiceStatus.Active, "Service must be active.");
        uint256 paymentNumber = block.timestamp / (paymentFrequency * 1 days);
        require(!payments[paymentNumber], "Payment already made for this period.");
        payments[paymentNumber] = true;
        emit PaymentMade(msg.sender, serviceFee);
    }
    
    function deliverService(string memory deliveryDetails) public {
        require(msg.sender == serviceProvider, "Only the service provider can deliver service.");
        require(status == ServiceStatus.Active, "Service must be active.");
        emit ServiceDelivered(msg.sender, deliveryDetails);
    }
    
    function activateService() public {
        require(msg.sender == serviceProvider, "Only the service provider can activate service.");
        require(status == ServiceStatus.Pending, "Service must be pending.");
        status = ServiceStatus.Active;
    }
    
    function completeService() public {
        require(msg.sender == serviceProvider, "Only the service provider can complete service.");
        require(status == ServiceStatus.Active, "Service must be active.");
        status = ServiceStatus.Completed;
    }
    
    function terminateAgreement() public {
        require(msg.sender == serviceProvider || msg.sender == client, "Unauthorized");
        require(block.timestamp < agreementEndDate, "Agreement already ended");
        status = ServiceStatus.Terminated;
        agreementEndDate = block.timestamp;
        emit AgreementTerminated(msg.sender);
    }
}