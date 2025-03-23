// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SaleAgreement {
    address public seller;
    address public buyer;
    string public itemDescription;
    uint256 public salePrice;
    uint256 public agreementEndDate;
    bool public isCompleted;
    
    enum SaleStatus { Pending, Locked, Completed, Cancelled }
    SaleStatus public status;
    
    event PaymentReceived(address indexed sender, uint256 amount);
    event ItemTransferred(address indexed seller, address indexed buyer);
    event AgreementTerminated(address indexed sender);
    event EscrowLocked(address indexed sender);
    event EscrowReleased(address indexed sender);
    
    constructor(
        address _seller,
        address _buyer,
        string memory _itemDescription,
        uint256 _salePrice,
        uint256 _agreementEndDate
    ) {
        seller = _seller;
        buyer = _buyer;
        itemDescription = _itemDescription;
        salePrice = _salePrice;
        agreementEndDate = _agreementEndDate;
        status = SaleStatus.Pending;
        isCompleted = false;
    }
    
    function makePayment() public {
        require(msg.sender == buyer, "Only the buyer can make payments.");
        require(block.timestamp < agreementEndDate, "Agreement has expired.");
        require(status == SaleStatus.Pending, "Sale must be in pending status.");
        status = SaleStatus.Locked;
        emit PaymentReceived(msg.sender, salePrice);
        emit EscrowLocked(msg.sender);
    }
    
    function confirmTransfer() public {
        require(msg.sender == seller, "Only the seller can confirm transfer.");
        require(status == SaleStatus.Locked, "Payment must be locked in escrow.");
        status = SaleStatus.Completed;
        isCompleted = true;
        emit ItemTransferred(seller, buyer);
        emit EscrowReleased(msg.sender);
    }
    
    function cancelAgreement() public {
        require(msg.sender == seller || msg.sender == buyer, "Unauthorized");
        require(block.timestamp < agreementEndDate, "Agreement has expired");
        require(status != SaleStatus.Completed, "Cannot cancel completed sale");
        status = SaleStatus.Cancelled;
        agreementEndDate = block.timestamp;
        emit AgreementTerminated(msg.sender);
    }
    
    function getStatus() public view returns (SaleStatus) {
        return status;
    }
}