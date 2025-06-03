import subprocess
from typing import TypedDict, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestConfig:
    test_dir: Path
    contract_name: str = "Agreement"
    solidity_version: str = "0.8.20"
    wait_for_human_review: bool = True


class TestResult(TypedDict):
    success: bool
    output: str
    test_file_path: Optional[str]


class TestGenerator:
    def __init__(self, config: TestConfig):
        self.config = config
        self._setup_test_environment()

    def _setup_test_environment(self) -> None:
        self.config.test_dir.mkdir(parents=True, exist_ok=True)
        print("Setting up test environment...")
        config_content = f"""
            require("@nomicfoundation/hardhat-toolbox");

            module.exports = {{
                solidity: "{self.config.solidity_version}",
                paths: {{
                    root: ".",
                    sources: "./contracts",
                    tests: "./test",
                    cache: "./cache",
                    artifacts: "./artifacts"
                }},
                networks: {{
                    hardhat: {{
                        // Set initial timestamp to a fixed value
                        chainId: 31337,
                        mining: {{
                            auto: true,
                            interval: 0
                        }}
                    }}
                }},
                mocha: {{
                    timeout: 40000
                }}
            }};
        """

        # Create contracts directory
        contracts_dir = self.config.test_dir / "contracts"
        contracts_dir.mkdir(exist_ok=True)

        (self.config.test_dir / "hardhat.config.js").write_text(config_content)

        subprocess.run(["npm", "init", "-y"],
                       cwd=self.config.test_dir, check=True)
        subprocess.run(
            ["npm", "install", "--save-dev", "hardhat",
                "@nomicfoundation/hardhat-toolbox"],
            cwd=self.config.test_dir,
            check=True
        )

    def generate_test_prompt(self, contract_code: str, agreement_text: str) -> str:
        return f"""
        You are a smart contract testing expert. Create comprehensive test cases for the following Solidity contract:

        Contract Code:
        {contract_code}

        Original Agreement:
        {agreement_text}

        Create a test file that follows these specific requirements:

        1. Test Setup:
           - Use Hardhat and Chai for testing
           - Deploy the contract in beforeEach using ethers.js
           - Get signers (owner, tenant, etc.) using ethers.getSigners()
           - Reset the network timestamp before each test
           - Example setup:
             ```javascript
             let contract;
             let owner;
             let tenant;
             
             beforeEach(async function () {{
                 // Reset network to a known state
                 await network.provider.send("evm_setNextBlockTimestamp", [Math.floor(Date.now() / 1000)]);
                 await network.provider.send("evm_mine");
                 
                 [owner, tenant] = await ethers.getSigners();
                 const Contract = await ethers.getContractFactory("Agreement");
                 contract = await Contract.deploy();
                 await contract.waitForDeployment();
             }});
             ```

        2. Test Categories:
           a. State Variables:
              - Test all state variables are initialized with correct values
              - Example:
                ```javascript
                it("Should initialize state variables correctly", async function () {{
                    expect(await contract.rentAmount()).to.equal(ethers.parseEther("1000"));
                    expect(await contract.startDate()).to.equal(startTimestamp);
                }});
                ```

           b. Access Control:
              - Test onlyOwner modifiers
              - Test role-based access control
              - Example:
                ```javascript
                it("Should only allow owner to modify critical parameters", async function () {{
                    await expect(contract.connect(tenant).setRentAmount(2000))
                        .to.be.revertedWith("Ownable: caller is not the owner");
                }});
                ```

           c. Payment Functions:
              - Test payment functions with correct amounts
              - Test payment functions with incorrect amounts
              - Test payment events
              - Example:
                ```javascript
                it("Should accept correct rent payment", async function () {{
                    await expect(contract.connect(tenant).payRent({{ value: ethers.parseEther("1000") }}))
                        .to.emit(contract, "RentPaid")
                        .withArgs(tenant.address, ethers.parseEther("1000"));
                }});
                ```

           d. Edge Cases:
              - Test zero values
              - Test maximum values
              - Test invalid inputs
              - Example:
                ```javascript
                it("Should revert when paying incorrect amount", async function () {{
                    await expect(contract.connect(tenant).payRent({{ value: ethers.parseEther("500") }}))
                        .to.be.revertedWith("Incorrect payment amount");
                }});
                ```

        3. Test Structure:
           - Each test should have a clear description
           - Use async/await for all contract interactions
           - Include proper error handling
           - Test both success and failure cases
           - Use proper assertions (expect, assert)

        4. Best Practices:
           - Use descriptive test names
           - Group related tests using describe blocks
           - Clean up any test state in afterEach if needed
           - Use proper error messages in assertions

        Output only the JavaScript test code using Hardhat and Chai, following the structure above.
        """

    def save_contract(self, contract_code: str) -> Path:
        contract_path = self.config.test_dir / "contracts" / \
            f"{self.config.contract_name}.sol"
        contract_path.write_text(contract_code)
        return contract_path

    def save_test_file(self, test_code: str) -> Path:
        test_dir = self.config.test_dir / "test"
        test_dir.mkdir(exist_ok=True)
        test_path = test_dir / f"{self.config.contract_name}.test.js"
        test_path.write_text(test_code)
        return test_path

    def wait_for_human_review(self, contract_path: Path, test_path: Path) -> bool:
        if not self.config.wait_for_human_review:
            return True

        print(f"\nContract saved to: {contract_path}")
        print(f"Test file saved to: {test_path}")
        print("\nPlease review and modify the files if needed.")
        input("Press Enter to continue with testing...")
        return True

    def run_tests(self) -> TestResult:
        try:
            result = subprocess.run(
                ["npx", "hardhat", "test"],
                cwd=self.config.test_dir,
                capture_output=True,
                text=True,
                check=True
            )
            return TestResult(
                success=True,
                output=result.stdout,
                test_file_path=str(
                    self.config.test_dir / "test" / f"{self.config.contract_name}.test.js")
            )
        except subprocess.CalledProcessError as e:
            return TestResult(
                success=False,
                output=e.stdout + e.stderr,
                test_file_path=str(
                    self.config.test_dir / "test" / f"{self.config.contract_name}.test.js")
            )

    def cleanup(self) -> None:
        import shutil
        shutil.rmtree(self.config.test_dir)
