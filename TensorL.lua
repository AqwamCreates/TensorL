--[[

	--------------------------------------------------------------------

	Version 1.0.0

	Aqwam's 3D Tensor Library (TensorL3D)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	By using or possesing any copies of this library, you agree to our license at:
	
	https://github.com/AqwamCreates/TensorL3D/LICENSE.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT WITHOUT PROPER PERMISSION!
	
	--------------------------------------------------------------------

--]]

local TensorL3D = {}

local function create3DTensor(dimensionArray, initialValue)
	
	local result = {}

	for dimension1 = 1, dimensionArray[1], 1 do

		result[dimension1] =  {}

		for dimension2 = 1, dimensionArray[2], 1 do

			result[dimension1][dimension2] = table.create(dimensionArray[3], initialValue)

		end

	end
	
	return result
	
end

local function create3DTensorFromFunction(dimensionArray, functionToUse)

	local result = {}

	for dimension1 = 1, dimensionArray[1], 1 do

		result[dimension1] =  {}

		for dimension2 = 1, dimensionArray[2], 1 do
			
			result[dimension1][dimension2] =  {}
			
			for dimension3 = 1, dimensionArray[3], 1 do
				
				result[dimension1][dimension2][dimension3] = functionToUse(dimension1, dimension2, dimension3)
				
			end

		end

	end

	return result

end

local function deepCopyTable(original, copies)

	copies = copies or {}

	local originalType = type(original)

	local copy

	if (originalType == 'table') then

		if copies[original] then

			copy = copies[original]

		else

			copy = {}

			copies[original] = copy

			for originalKey, originalValue in next, original, nil do

				copy[deepCopyTable(originalKey, copies)] = deepCopyTable(originalValue, copies)

			end

			setmetatable(copy, deepCopyTable(getmetatable(original), copies))

		end

	else -- number, string, boolean, etc

		copy = original

	end

	return copy

end

local function applyFunction(functionToApply, tensor1, tensor2)
	
	local result = {}
	
	for dimension1 = 1, #tensor1, 1 do
		
		result[dimension1] = {}

		for dimension2 = 1, #tensor1[dimension1], 1 do
			
			result[dimension1][dimension2] = {}

			for dimension3 = 1, #tensor1[dimension1][dimension2], 1 do

				result[dimension1][dimension2][dimension3] = functionToApply(tensor1[dimension1][dimension2][dimension3], tensor2[dimension1][dimension2][dimension3]) 

			end

		end

	end
	
	return result
	
end

local function generateTensor2DString(tensor2D)

	if tensor2D == nil then return "" end

	local numberOfRows = #tensor2D

	local numberOfColumns = #tensor2D[1]

	local columnWidths = {}

	for column = 1, numberOfColumns do

		local maxWidth = 0

		for row = 1, numberOfRows do

			local cellWidth = string.len(tostring(tensor2D[row][column]))

			if (cellWidth > maxWidth) then

				maxWidth = cellWidth

			end

		end

		columnWidths[column] = maxWidth

	end

	local text = ""

	for row = 1, numberOfRows do

		text = text .. "{"

		for column = 1, numberOfColumns do

			local cellValue = tensor2D[row][column]

			local cellText = tostring(cellValue)

			local cellWidth = string.len(cellText)

			local padding = columnWidths[column] - cellWidth + 1

			text = text .. string.rep(" ", padding) .. cellText
		end

		text = text .. " }\n"

	end

	return text

end

local function sum(tensor, dimension)
	
	local dimensionArray = tensor:getSize()

	local newDimensionArray = deepCopyTable(dimensionArray)

	if dimension then newDimensionArray[dimension] = 1 end

	local result = (not dimension and 0) or TensorL3D.create(newDimensionArray, 0)

	for dimension1 = 1, dimensionArray[1], 1 do

		for dimension2 = 1, dimensionArray[2], 1 do

			for dimension3 = 1, dimensionArray[3], 1 do

				if (dimension == nil) then

					result += tensor[dimension1][dimension2][dimension3]

				elseif (dimension == 1) then

					result[1][dimension2][dimension3] += tensor[dimension1][dimension2][dimension3]	

				elseif (dimension == 2) then

					result[dimension1][1][dimension3] += tensor[dimension1][dimension2][dimension3]

				elseif (dimension == 3) then

					result[dimension1][dimension2][1] += tensor[dimension1][dimension2][dimension3]

				else

					error("Invalid dimension.")

				end 

			end

		end	

	end
	
	return result
	
end

local function onBroadcastError(tensor1, tensor2)

	local errorMessage = "Unable To Broadcast. \n" .. "Tensor 1 Size: " .. "(" .. #tensor1 .. ", " .. #tensor1[1] .. ", " .. #tensor1[1][1] .. ") \n" .. "Tensor 2 Size: " .. "(" .. #tensor2[1] .. ", " .. #tensor2[1] .. ", " .. #tensor2[1][1] .. ") \n"

	error(errorMessage)

end

local function checkIfCanBroadcast(tensor1, tensor2)

	local tensor1Depth = #tensor1
	local tensor2Depth = #tensor2

	local tensor1Rows = #tensor1[1]
	local tensor2Rows = #tensor2[1]

	local tensor1Columns = #tensor1[1][1]
	local tensor2Columns = #tensor2[1][1]

	local isTensor1Broadcasted
	local isTensor2Broadcasted

	local hasSameRowSize = (tensor1Rows == tensor2Rows)
	local hasSameColumnSize = (tensor1Columns == tensor2Columns)
	local hasSameDepth = (tensor1Depth == tensor2Depth)

	local hasSameDimension = hasSameRowSize and hasSameColumnSize and hasSameDepth

	local isTensor1LargerInOneDimension = ((tensor1Depth > 1) and hasSameRowSize and hasSameColumnSize and (tensor2Depth == 1)) or
		((tensor1Rows > 1) and hasSameColumnSize and hasSameDepth and (tensor2Rows == 1)) or
		((tensor1Columns > 1) and hasSameRowSize and hasSameDepth and (tensor2Columns == 1))

	local isTensor2LargerInOneDimension = ((tensor2Depth > 1) and hasSameRowSize and hasSameColumnSize and (tensor1Depth == 1)) or
		((tensor2Rows > 1) and hasSameColumnSize and hasSameDepth and (tensor1Rows == 1)) or
		((tensor2Columns > 1) and hasSameRowSize and hasSameDepth and (tensor1Columns == 1))

	local isTensor1Scalar = (tensor1Depth == 1) and (tensor1Rows == 1) and (tensor1Columns == 1)
	local isTensor2Scalar = (tensor2Depth == 1) and (tensor2Rows == 1) and (tensor2Columns == 1)

	local isTensor1Larger = (tensor1Depth > tensor2Depth) and (tensor1Rows > tensor2Rows) and (tensor1Columns > tensor2Columns)
	local isTensor2Larger = (tensor2Depth > tensor1Depth) and (tensor2Rows > tensor1Rows) and (tensor2Columns > tensor1Columns)

	if (hasSameDimension) then

		isTensor1Broadcasted = false
		isTensor2Broadcasted = false

	elseif (isTensor2LargerInOneDimension) or (isTensor2Larger and isTensor1Scalar) then

		isTensor1Broadcasted = true
		isTensor2Broadcasted = false

	elseif (isTensor1LargerInOneDimension) or (isTensor1Larger and isTensor2Scalar) then

		isTensor1Broadcasted = false
		isTensor2Broadcasted = true

	else

		onBroadcastError(tensor1, tensor2)

	end

	return isTensor1Broadcasted, isTensor2Broadcasted

end

local function expandTensor(tensor, targetDepthSize, targetRowSize, targetColumnSize)

	local isDepthSizeEqualToOne = (#tensor == 1)

	local isRowSizeEqualToOne = (#tensor[1] == 1)

	local isColumnSizeEqualToOne = (#tensor[1][1] == 1)

	local result = {}

	if (isDepthSizeEqualToOne) and (isRowSizeEqualToOne) and (isColumnSizeEqualToOne) then

		for i = 1, targetDepthSize do

			result[i] = {}

			for j = 1, targetRowSize do

				result[i][j] = {}

				for k = 1, targetColumnSize do

					result[i][j][k] = tensor[1][1][1]

				end

			end

		end

	elseif (not isDepthSizeEqualToOne) and (isRowSizeEqualToOne) and (isColumnSizeEqualToOne) then

		for i = 1, targetDepthSize do

			result[i] = {}

			for j = 1, targetRowSize do

				result[i][j] = {}

				for k = 1, targetColumnSize do

					result[i][j][k] = tensor[i][1][1]

				end

			end

		end

	elseif (not isDepthSizeEqualToOne) and (not isRowSizeEqualToOne) and (isColumnSizeEqualToOne) then

		for i = 1, targetDepthSize do

			result[i] = {}

			for j = 1, targetRowSize do

				result[i][j] = {}

				for k = 1, targetColumnSize do

					result[i][j][k] = tensor[i][j][1]

				end

			end

		end

	elseif (isDepthSizeEqualToOne) and (not isRowSizeEqualToOne) and (isColumnSizeEqualToOne) then

		for i = 1, targetDepthSize do

			result[i] = {}

			for j = 1, targetRowSize do

				result[i][j] = {}

				for k = 1, targetColumnSize do

					result[i][j][k] = tensor[1][j][1]

				end

			end

		end

	elseif (isDepthSizeEqualToOne) and (isRowSizeEqualToOne) and (not isColumnSizeEqualToOne) then

		for i = 1, targetDepthSize do

			result[i] = {}

			for j = 1, targetRowSize do

				result[i][j] = {}

				for k = 1, targetColumnSize do

					result[i][j][k] = tensor[1][1][k]

				end

			end

		end

	elseif (not isDepthSizeEqualToOne) and (isRowSizeEqualToOne) and (not isColumnSizeEqualToOne) then

		for i = 1, targetDepthSize do

			result[i] = {}

			for j = 1, targetRowSize do

				result[i][j] = {}

				for k = 1, targetColumnSize do

					result[i][j][k] = tensor[i][1][k]

				end

			end

		end

	elseif (isDepthSizeEqualToOne) and (not isRowSizeEqualToOne) and (not isColumnSizeEqualToOne) then

		for i = 1, targetDepthSize do

			result[i] = {}

			for j = 1, targetRowSize do

				result[i][j] = {}

				for k = 1, targetColumnSize do

					result[i][j][k] = tensor[1][j][k]

				end

			end

		end

	elseif (not isDepthSizeEqualToOne) and (not isRowSizeEqualToOne) and (not isColumnSizeEqualToOne) then

		result = tensor

	end

	return result

end

local function broadcastTensorsIfDifferentSizes(tensor1, tensor2)

	local isTensor1Broadcasted = false
	local isTensor2Broadcasted = false

	isTensor1Broadcasted, isTensor2Broadcasted = checkIfCanBroadcast(tensor1, tensor2)

	if (isTensor1Broadcasted) then

		tensor1 = expandTensor(tensor1, #tensor2, #tensor2[1], #tensor2[1][1])

	elseif (isTensor2Broadcasted) then

		tensor2 = expandTensor(tensor2, #tensor1, #tensor1[1], #tensor1[1][1])

	end

	return tensor1, tensor2

end

local function is3DTensor(tensor)

	local isTensor = pcall(function() local _ = tensor[1][1][1] end)

	return isTensor

end

local function convertValueTo3DTensor(value)
	
	if is3DTensor(value) then return value end

	if (type(value) ~= "number") then error("Cannot convert value into 3D tensor.") end
	
	return {{{value}}}
	
end

local function isDimensionArrayEqual(dimensionArray, otherDimensionArray)
	
	for index, _ in ipairs(dimensionArray) do if (dimensionArray[index] ~= otherDimensionArray[index]) then return false end end
	
	return true
	
end

local function throwErrorIfOtherValueIsNot3DTensor(otherTensor)

	if not is3DTensor(otherTensor) then return error("The other value is not a 3D tensor.") end

end

local function throwErrorIfValueIsNot3DTensor(otherTensor)

	if not is3DTensor(otherTensor) then return error("The value is not a 3D tensor.") end

end

local function throwErrorIfDimensionArrayLengthIsNotEqualToThree(dimensionArray)
	
	if (#dimensionArray ~= 3) then return error("The length of dimension array is not equal to 3.") end
	
end

local function throwErrorIfDimensionArrayIsNotEqual(dimensionArray, otherDimensionArray)
	
	if not isDimensionArrayEqual(dimensionArray, otherDimensionArray) then error("The values of dimension array are not equal.") end
	
end



function TensorL3D.new(value)
	
	throwErrorIfValueIsNot3DTensor(value)
	
	local self = setmetatable({}, TensorL3D)

	self.Values = value

	return self
	
end

function TensorL3D.create(dimensionArray, initialValue)
	
	throwErrorIfDimensionArrayLengthIsNotEqualToThree(dimensionArray)
	
	initialValue = initialValue or 0
	
	local self = setmetatable({}, TensorL3D)
	
	self.Values = create3DTensor(dimensionArray, initialValue)
	
	return self
	
end

function TensorL3D.createFromFunction(dimensionArray, functionToUse)
	
	throwErrorIfDimensionArrayLengthIsNotEqualToThree(dimensionArray)
	
	if (type(functionToUse) == "nil") then error("No function.") end
	
	local self = setmetatable({}, TensorL3D)

	self.Values = create3DTensorFromFunction(dimensionArray, functionToUse)

	return self
	
end


function TensorL3D:expand(dimensionArray)
	
	throwErrorIfDimensionArrayLengthIsNotEqualToThree(dimensionArray)
	
	local newTensor = expandTensor(self, dimensionArray[1], dimensionArray[2], dimensionArray[3])

	return self.new(newTensor)

end

function TensorL3D:getSize()
	
	return {#self, #self[1], #self[1][1]}
	
end

function TensorL3D:print()

	print(self)
	
end

function TensorL3D:transpose(dimensionIndexArray)
	
	if (#dimensionIndexArray ~= 2) then error("The length of dimension index array is not equal to 2.") end
	
	local dimension1 = dimensionIndexArray[1]
	
	local dimension2 = dimensionIndexArray[2]
	
	if (type(dimension1) ~= "number") or (type(dimension2) ~= "number") then error("Dimensions are not numbers.") end
	
	if (dimension1 <= 0) or (dimension1 >= 4) or (dimension2 <= 0) or (dimension2 >= 4) or (dimension1 == dimension2) then
		
		error("Invalid dimensions for transpose.")
		
	end
	
	local newDimensionArray = self:getSize()

	newDimensionArray[dimension1], newDimensionArray[dimension2] = newDimensionArray[dimension2], newDimensionArray[dimension1]
	
	local newTensor = self.create(newDimensionArray, true)
	
	if (table.find(dimensionIndexArray, 1)) and (table.find(dimensionIndexArray, 2)) then
		
		for i = 1, newDimensionArray[1] do
			
			for j = 1, newDimensionArray[2] do
				
				for k = 1, newDimensionArray[3] do
					
					newTensor[i][j][k] = self[j][i][k]
					
				end
				
			end
			
		end
		
	elseif (table.find(dimensionIndexArray, 1)) and (table.find(dimensionIndexArray, 3)) then
		
		for i = 1, newDimensionArray[1] do

			for j = 1, newDimensionArray[2] do

				for k = 1, newDimensionArray[3] do

					newTensor[i][j][k] = self[k][j][i]

				end

			end

		end
		
	elseif (table.find(dimensionIndexArray, 2)) and (table.find(dimensionIndexArray, 3)) then
		
		for i = 1, newDimensionArray[1] do

			for j = 1, newDimensionArray[2] do

				for k = 1, newDimensionArray[3] do

					newTensor[i][j][k] = self[i][k][j]

				end

			end

		end
		
	end

	return newTensor
	
end

function TensorL3D:__eq(other)
	
	if not is3DTensor(other) then return false end
	
	for dimension1 = 1, #self, 1 do

		for dimension2 = 1, #self[dimension1], 1 do

			for dimension3 = 1, #self[dimension1][dimension2], 1 do

				if (self[dimension1][dimension2][dimension3] ~= other[dimension1][dimension2][dimension3]) then return false end

			end

		end

	end

	return true
	
end

function TensorL3D:isEqualTo(other)
	
	throwErrorIfOtherValueIsNot3DTensor(other)

	local functionToApply = function(a, b) return (a == b) end

	local result = applyFunction(functionToApply, self, other)

	return self.new(result)

end

function TensorL3D:isGreaterThan(other)

	throwErrorIfOtherValueIsNot3DTensor(other)

	local functionToApply = function(a, b) return (a > b) end

	local result = applyFunction(functionToApply, self, other)

	return self.new(result)

end

function TensorL3D:isGreaterOrEqualTo(other)

	throwErrorIfOtherValueIsNot3DTensor(other)

	local functionToApply = function(a, b) return (a >= b) end

	local result = applyFunction(functionToApply, self, other)

	return self.new(result)

end

function TensorL3D:isLessThan(other)

	throwErrorIfOtherValueIsNot3DTensor(other)
	
	local functionToApply = function(a, b) return (a < b) end

	local result = applyFunction(functionToApply, self, other)
	
	return self.new(result)

end

function TensorL3D:isLessOrEqualTo(other)

	throwErrorIfOtherValueIsNot3DTensor(other)

	local functionToApply = function(a, b) return (a <= b) end

	local result = applyFunction(functionToApply, self, other)

	return self.new(result)

end

function TensorL3D:sum(dimension)
	
	return sum(self, dimension)
	
end

function TensorL3D:concatenate(other, dimension)

	throwErrorIfOtherValueIsNot3DTensor(other)

	local dimensionArray1 = self:getSize()

	local dimensionArray2 = other:getSize()
	
	local newDimensionArray = {}
	
	for dimensionIndex = 1, 3, 1 do
		
		if (dimensionIndex == dimension) then continue end
		
		if (dimensionArray1[dimensionIndex] ~= dimensionArray2[dimension]) then error("The tensors do not contain equal dimension values at dimension " .. dimension .. ".") end
		
	end
	
	for dimensionIndex = 1, 3, 1 do
		
		local dimensionSize = dimensionArray1[dimensionIndex]
		
		if (dimensionIndex == dimension) then
			
			dimensionSize = dimensionSize + dimensionArray2[dimensionIndex]
			
		end
		
		table.insert(newDimensionArray, dimensionSize)
		
	end
	
	local newTensor = self.create(newDimensionArray, true)
	
	
	return newTensor
	
end


function TensorL3D:dotProduct(other, dimension)
	
	throwErrorIfOtherValueIsNot3DTensor(other)
	
	local dimensionArray1 = self:getSize()
	
	local dimensionArray2 = other:getSize()
	
	if (dimensionArray1[dimension] ~= dimensionArray2[dimension]) then error("The tensors do not contain equal dimension values at dimension " .. dimension .. ".") end
	
	local functionToApply = function(a, b) return (a * b) end

	local result = applyFunction(functionToApply, self, other)

	result = TensorL3D:sum(result, dimension)
	
	return result
	
end

function TensorL3D:innerProduct(other)

	other = convertValueTo3DTensor(other)
	
	throwErrorIfOtherValueIsNot3DTensor(other)

	local functionToApply = function(a, b) return (a * b) end

	local result = applyFunction(functionToApply, self, other)
	
	result = self.new(result)
	
	result = result:sum(1)
	
	result = result:sum(2)
	
	result = result:sum(3)
	
	return result[1][1][1]

end

function TensorL3D:copy()
	
	return deepCopyTable(self)
	
end

function TensorL3D:rawCopy()

	return deepCopyTable(self.Values)

end

function TensorL3D:applyFunction(functionToApply, ...)

	local tensorValues

	local tensors = {...}
	
	local dimensionArray = self:getSize()

	local result = self.create(dimensionArray)

	for dimension1 = 1, dimensionArray[1], 1 do

		for dimension2 = 1, dimensionArray[2], 1 do

			for dimension3 = 1, dimensionArray[3], 1 do
				
				tensorValues = {}
				
				for tensorIndex = 1, #tensors, 1  do
					
					table.insert(tensorValues, tensors[tensorIndex][dimension1][dimension2][dimension3])

				end 
				
				result[dimension1][dimension2][dimension3] = functionToApply(self, table.unpack(tensorValues))
				
			end
			
		end	

	end

	return result

end

function TensorL3D:__add(other)
	
	other = convertValueTo3DTensor(other)
	
	throwErrorIfOtherValueIsNot3DTensor(other)
	
	local newSelf, newOther = broadcastTensorsIfDifferentSizes(self, other)
	
	local functionToApply = function(a, b) return (a + b) end
	
	local result = applyFunction(functionToApply, newSelf, newOther)

	return self.new(result)
	
end

function TensorL3D:__sub(other)
	
	other = convertValueTo3DTensor(other)
	
	throwErrorIfOtherValueIsNot3DTensor(other)
	
	local newSelf, newOther = broadcastTensorsIfDifferentSizes(self, other)

	local functionToApply = function(a, b) return (a - b) end

	local result = applyFunction(functionToApply, newSelf, newOther)

	return self.new(result)
	
end

function TensorL3D:__mul(other)
	
	other = convertValueTo3DTensor(other)
	
	throwErrorIfOtherValueIsNot3DTensor(other)
	
	local newSelf, newOther = broadcastTensorsIfDifferentSizes(self, other)

	local functionToApply = function(a, b) return (a * b) end

	local result = applyFunction(functionToApply, newSelf, newOther)

	return self.new(result)
	
end

function TensorL3D:__div(other)
	
	other = convertValueTo3DTensor(other)
	
	throwErrorIfOtherValueIsNot3DTensor(other)
	
	local newSelf, newOther = broadcastTensorsIfDifferentSizes(self, other)

	local functionToApply = function(a, b) return (a / b) end

	local result = applyFunction(functionToApply, newSelf, newOther)

	return self.new(result)
	
end

function TensorL3D:__unm()
	
	local result = {}
	
	local dimensionArray = self:getSize()
	
	for dimension1 = 1, dimensionArray[1], 1 do
		
		result[dimension1] = {}

		for dimension2 = 1, dimensionArray[2], 1 do
			
			result[dimension1][dimension2] = {}
			
			for dimension3 = 1, dimensionArray[3], 1 do

				result[dimension1][dimension2][dimension3] = -self[dimension1][dimension2][dimension3]

			end

		end

	end
	
	return self.new(result)
	
end

function TensorL3D:__tostring()
	
	local text = "\n\n{\n\n"

	local generatedText

	for index = 1, #self, 1 do

		generatedText = generateTensor2DString(self[index])

		text = text .. generatedText .. "\n"

	end

	text = text .. "}\n\n"

	return text
	
end

function TensorL3D:__len()
	
	return #self.Values
	
end

function TensorL3D:__index(index)
	
	if (type(index) == "number") then
		
		return rawget(self.Values, index)
		
	else
		
		return rawget(TensorL3D, index)
		
	end
	
end

function TensorL3D:__newindex(index, value)
	
	rawset(self, index, value)
	
end

return TensorL3D
