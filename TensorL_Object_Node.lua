local AqwamTensorLibrary = {}

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

local function getDimensionSizeArrayRecursive(tensor, targetDimensionSizeArray)

	if (type(tensor) ~= "table") then return end

	table.insert(targetDimensionSizeArray, #tensor)

	getDimensionSizeArrayRecursive(tensor[1], targetDimensionSizeArray)

end

local function getDimensionSizeArray(tensor)

	local dimensionSizeArray = {}

	getDimensionSizeArrayRecursive(tensor, dimensionSizeArray)

	return dimensionSizeArray

end

local function applyFunctionUsingOneTensor(functionToApply, tensor, dimensionSizeArray) -- Dimension size array is put here because it is computationally expensive to use recurvsive just to get the dimension size.

	local numberOfDimensions = #dimensionSizeArray

	local resultTensor = {}

	if (numberOfDimensions >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do resultTensor[i] = applyFunctionUsingOneTensor(functionToApply, tensor[i], remainingDimensionSizeArray) end

	elseif (numberOfDimensions == 1) then -- Much more efficient than applying recursion again to get the original value.

		for i = 1, dimensionSizeArray[1], 1 do resultTensor[i] = functionToApply(tensor[i]) end

	else -- Sometimes the original tensor can be a number, so we must do the operation directly.

		resultTensor = functionToApply(tensor)

	end

	return resultTensor

end

local function applyFunctionUsingTwoTensors(functionToApply, tensor1, tensor2, dimensionSizeArray) -- Dimension size array is put here because it is computationally expensive to use recurvsive just to get the dimension size.

	local numberOfDimensions = #dimensionSizeArray

	local resultTensor = {}

	if (numberOfDimensions >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do resultTensor[i] = applyFunctionUsingTwoTensors(functionToApply, tensor1[i], tensor2[i], remainingDimensionSizeArray) end

	elseif (numberOfDimensions == 1) then -- Much more efficient than applying recursion again to get the original value.

		for i = 1, dimensionSizeArray[1], 1 do resultTensor[i] = functionToApply(tensor1[i], tensor2[i]) end

	else -- Sometimes the original tensor can be a number, so we must do the operation directly.

		resultTensor = functionToApply(tensor1, tensor2)

	end

	return resultTensor

end

local function applyFunctionWhenTheFirstValueIsAScalar(functionToApply, scalar, tensor, dimensionSizeArray) -- Dimension size array is put here because it is computationally expensive to use recurvsive just to get the dimension size.

	local numberOfDimensions = #dimensionSizeArray

	local resultTensor = {}

	if (numberOfDimensions >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do resultTensor[i] = applyFunctionWhenTheFirstValueIsAScalar(functionToApply, scalar, tensor[i], remainingDimensionSizeArray) end

	elseif (numberOfDimensions == 1) then -- Much more efficient than applying recursion again to get the original value.

		for i = 1, dimensionSizeArray[1], 1 do resultTensor[i] = functionToApply(scalar, tensor[i]) end

	else -- Sometimes the original tensor can be a number, so we must do the operation directly.

		resultTensor = functionToApply(scalar, tensor)

	end

	return resultTensor

end

local function applyFunctionWhenTheSecondValueIsAScalar(functionToApply, tensor, scalar, dimensionSizeArray) -- Dimension size array is put here because it is computationally expensive to use recurvsive just to get the dimension size.

	local numberOfDimensions = #dimensionSizeArray

	local resultTensor = {}

	if (numberOfDimensions >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do resultTensor[i] = applyFunctionWhenTheSecondValueIsAScalar(functionToApply, tensor[i], scalar, remainingDimensionSizeArray) end

	elseif (numberOfDimensions == 1) then -- Much more efficient than applying recursion again to get the original value.

		for i = 1, dimensionSizeArray[1], 1 do resultTensor[i] = functionToApply(tensor[i], scalar) end

	else -- Sometimes the original tensor can be a number, so we must do the operation directly.

		resultTensor = functionToApply(tensor, scalar)

	end

	return resultTensor

end

local function applyFunctionOnMultipleTensors(functionToApply, ...)

	local tensorArray = {...}

	local numberOfTensors = #tensorArray

	local tensor = tensorArray[1]

	if (numberOfTensors == 1) then 

		local dimensionSizeArray = getDimensionSizeArray(tensor)

		if (type(tensor) == "table") then

			return applyFunctionUsingOneTensor(functionToApply, tensor, dimensionSizeArray)

		else

			return functionToApply(tensor)

		end

	end

	for i = 2, numberOfTensors, 1 do

		local otherTensor = tensorArray[i]

		local isFirstValueATensor = (type(tensor) == "table")

		local isSecondValueATensor = (type(otherTensor) == "table")

		if (isFirstValueATensor) and (isSecondValueATensor) then

			tensor, otherTensor = broadcast(tensor, otherTensor, false)

			local dimensionSizeArray = getDimensionSizeArray(tensor)

			tensor = applyFunctionUsingTwoTensors(functionToApply, tensor, otherTensor, dimensionSizeArray)

		elseif (not isFirstValueATensor) and (isSecondValueATensor) then

			local dimensionSizeArray = getDimensionSizeArray(otherTensor)

			tensor = applyFunctionWhenTheFirstValueIsAScalar(functionToApply, tensor, otherTensor, dimensionSizeArray)

		elseif (isFirstValueATensor) and (not isSecondValueATensor) then

			local dimensionSizeArray = getDimensionSizeArray(tensor)

			tensor = applyFunctionWhenTheSecondValueIsAScalar(functionToApply, tensor, otherTensor, dimensionSizeArray)

		else

			tensor = functionToApply(tensor, otherTensor)

		end

	end

	return tensor

end

function AqwamTensorLibrary.new(tensor)

	local self = setmetatable({}, AqwamTensorLibrary)
	
	local tensorToBeStored
	
	if (type(tensor[1]) == "number") then
		
		tensorToBeStored = tensor
		
	else
		
		tensorToBeStored = {}
		
		for i, subTensor in ipairs(tensor) do tensorToBeStored[i] = AqwamTensorLibrary.new(subTensor) end
		
	end
	
	self.tensor = tensorToBeStored
	
	return self

end

function AqwamTensorLibrary:__len()

	return #self.tensor

end

function AqwamTensorLibrary:__index(index)

	if (typeof(index) == "number") then

		return rawget(self.tensor, index)

	else

		return rawget(AqwamTensorLibrary, index)

	end

end

function AqwamTensorLibrary:__newindex(index, value)

	rawset(self, index, value)

end

function AqwamTensorLibrary:getDimensionSizeArray()
	
	return getDimensionSizeArray(self)
	
end

local function getNumberOfDimensions(tensor)

	if (type(tensor) ~= "table") then return 0 end

	return getNumberOfDimensions(tensor[1]) + 1

end

function AqwamTensorLibrary:getNumberOfDimensions()
	
	return getNumberOfDimensions(self)
	
end

function AqwamTensorLibrary:copy()

	return deepCopyTable(self)

end

function AqwamTensorLibrary:rawCopy()

	return deepCopyTable(self.tensor)

end

function AqwamTensorLibrary:__add(other)

	local resultTensor = applyFunctionOnMultipleTensors(function(a, b) return (a + b) end, self, other)

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:add(...)

	local functionToApply = function(a, b) return (a + b) end

	local resultTensor

	if (self.tensor) then

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, ...)

	end

	return resultTensor

end

function AqwamTensorLibrary:__sub(other)

	local resultTensor = applyFunctionOnMultipleTensors(function(a, b) return (a - b) end, self, other)

	return resultTensor

end

function AqwamTensorLibrary:subtract(...)

	local functionToApply = function(a, b) return (a - b) end

	local resultTensor

	if (self.tensor) then

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, ...)

	end

	return resultTensor

end

function AqwamTensorLibrary:__mul(other)

	local resultTensor = applyFunctionOnMultipleTensors(function(a, b) return (a * b) end, self, other)

	return resultTensor

end

function AqwamTensorLibrary:multiply(...)

	local functionToApply = function(a, b) return (a * b) end

	local resultTensor

	if (self.tensor) then

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, ...)

	end

	return resultTensor

end

function AqwamTensorLibrary:__div(other)

	local resultTensor = applyFunctionOnMultipleTensors(function(a, b) return (a / b) end, self, other)

	return resultTensor

end

function AqwamTensorLibrary:divide(...)

	local functionToApply = function(a, b) return (a / b) end

	local resultTensor

	if (self.tensor) then

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, ...)

	end

	return resultTensor

end

function AqwamTensorLibrary:__unm()

	local resultTensor = applyFunctionOnMultipleTensors(function(a) return (-a) end, self)

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:unaryMinus(...)

	local functionToApply = function(a) return (-a) end

	local resultTensor

	if (self.tensor) then

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, ...)

	end

	return resultTensor

end

return AqwamTensorLibrary
