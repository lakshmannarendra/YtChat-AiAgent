import { Command as $Command } from "@smithy/smithy-client";
import { MetadataBearer as __MetadataBearer } from "@smithy/types";
import { CloudFormationClientResolvedConfig, ServiceInputTypes, ServiceOutputTypes } from "../CloudFormationClient";
import { ListResourceScansInput, ListResourceScansOutput } from "../models/models_0";
/**
 * @public
 */
export type { __MetadataBearer };
export { $Command };
/**
 * @public
 *
 * The input for {@link ListResourceScansCommand}.
 */
export interface ListResourceScansCommandInput extends ListResourceScansInput {
}
/**
 * @public
 *
 * The output of {@link ListResourceScansCommand}.
 */
export interface ListResourceScansCommandOutput extends ListResourceScansOutput, __MetadataBearer {
}
declare const ListResourceScansCommand_base: {
    new (input: ListResourceScansCommandInput): import("@smithy/smithy-client").CommandImpl<ListResourceScansCommandInput, ListResourceScansCommandOutput, CloudFormationClientResolvedConfig, ServiceInputTypes, ServiceOutputTypes>;
    new (...[input]: [] | [ListResourceScansCommandInput]): import("@smithy/smithy-client").CommandImpl<ListResourceScansCommandInput, ListResourceScansCommandOutput, CloudFormationClientResolvedConfig, ServiceInputTypes, ServiceOutputTypes>;
    getEndpointParameterInstructions(): import("@smithy/middleware-endpoint").EndpointParameterInstructions;
};
/**
 * <p>List the resource scans from newest to oldest. By default it will return up to 10 resource
 *       scans.</p>
 * @example
 * Use a bare-bones client and the command you need to make an API call.
 * ```javascript
 * import { CloudFormationClient, ListResourceScansCommand } from "@aws-sdk/client-cloudformation"; // ES Modules import
 * // const { CloudFormationClient, ListResourceScansCommand } = require("@aws-sdk/client-cloudformation"); // CommonJS import
 * const client = new CloudFormationClient(config);
 * const input = { // ListResourceScansInput
 *   NextToken: "STRING_VALUE",
 *   MaxResults: Number("int"),
 *   ScanTypeFilter: "FULL" || "PARTIAL",
 * };
 * const command = new ListResourceScansCommand(input);
 * const response = await client.send(command);
 * // { // ListResourceScansOutput
 * //   ResourceScanSummaries: [ // ResourceScanSummaries
 * //     { // ResourceScanSummary
 * //       ResourceScanId: "STRING_VALUE",
 * //       Status: "IN_PROGRESS" || "FAILED" || "COMPLETE" || "EXPIRED",
 * //       StatusReason: "STRING_VALUE",
 * //       StartTime: new Date("TIMESTAMP"),
 * //       EndTime: new Date("TIMESTAMP"),
 * //       PercentageCompleted: Number("double"),
 * //       ScanType: "FULL" || "PARTIAL",
 * //     },
 * //   ],
 * //   NextToken: "STRING_VALUE",
 * // };
 *
 * ```
 *
 * @param ListResourceScansCommandInput - {@link ListResourceScansCommandInput}
 * @returns {@link ListResourceScansCommandOutput}
 * @see {@link ListResourceScansCommandInput} for command's `input` shape.
 * @see {@link ListResourceScansCommandOutput} for command's `response` shape.
 * @see {@link CloudFormationClientResolvedConfig | config} for CloudFormationClient's `config` shape.
 *
 * @throws {@link CloudFormationServiceException}
 * <p>Base exception class for all service exceptions from CloudFormation service.</p>
 *
 *
 * @public
 */
export declare class ListResourceScansCommand extends ListResourceScansCommand_base {
    /** @internal type navigation helper, not in runtime. */
    protected static __types: {
        api: {
            input: ListResourceScansInput;
            output: ListResourceScansOutput;
        };
        sdk: {
            input: ListResourceScansCommandInput;
            output: ListResourceScansCommandOutput;
        };
    };
}
